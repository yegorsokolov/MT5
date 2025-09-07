"""Training routine for the Adaptive MT5 bot."""

from pathlib import Path
import random
import json
import logging
import argparse
import asyncio
import joblib
import pandas as pd
import math
import torch
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from analysis.purged_cv import PurgedTimeSeriesSplit
from lightgbm import LGBMClassifier
from analytics import mlflow_client as mlflow
from datetime import datetime

from data.feature_scaler import FeatureScaler

from log_utils import setup_logging, log_exceptions, LOG_DIR
import numpy as np
from risk.position_sizer import PositionSizer
from ray_utils import (
    init as ray_init,
    shutdown as ray_shutdown,
    cluster_available,
    submit,
)

try:
    import shap
except Exception:  # noqa: E722
    shap = None

from utils import load_config
from utils.resource_monitor import monitor
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
    load_history_iter,
)
from data.features import make_features
from data.labels import triple_barrier
from state_manager import save_checkpoint, load_latest_checkpoint
from analysis.regime_detection import periodic_reclassification
from models import model_store
from analysis.feature_selector import select_features
from analysis.prob_calibration import (
    ProbabilityCalibrator,
    CalibratedModel,
    log_reliability,
)
from analysis.active_learning import ActiveLearningQueue, merge_labels
from analysis import model_card
from analysis.domain_adapter import DomainAdapter
from models.conformal import fit_residuals, predict_interval, evaluate_coverage

setup_logging()
logger = logging.getLogger(__name__)

# start capability monitoring
monitor.start()

# Track active classifiers for dynamic resizing
_ACTIVE_CLFS: list[LGBMClassifier] = []


def _register_clf(clf: LGBMClassifier) -> None:
    """Keep reference to classifier for dynamic n_jobs updates."""
    _ACTIVE_CLFS.append(clf)


def _lgbm_params(cfg: dict) -> dict:
    """Extract LightGBM hyper-parameters from config."""
    return {k: cfg[k] for k in ("num_leaves", "learning_rate", "max_depth") if k in cfg}


def _subscribe_cpu_updates(cfg: dict) -> None:
    async def _watch() -> None:
        q = monitor.subscribe()
        while True:
            await q.get()
            n_jobs = cfg.get("n_jobs") or monitor.capabilities.cpus
            for c in list(_ACTIVE_CLFS):
                try:
                    c.set_params(n_jobs=n_jobs)
                except Exception:
                    logger.debug("Failed to update n_jobs for classifier")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    loop.create_task(_watch())


def _load_donor_booster(symbol: str):
    """Load LightGBM booster for the latest model trained on a symbol."""
    try:
        versions = model_store.list_versions()
    except Exception:  # pragma: no cover - store may not exist
        return None
    for meta in reversed(versions):
        cfg = meta.get("training_config", {})
        syms = cfg.get("symbols") or [cfg.get("symbol")]
        if syms and symbol in syms:
            model, _ = model_store.load_model(meta["version_id"])
            clf = (
                model.named_steps.get("clf") if hasattr(model, "named_steps") else None
            )
            if clf and hasattr(clf, "booster_"):
                return clf.booster_
    return None


def log_shap_importance(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    features: list[str],
    report_dir: Path | None = None,
) -> None:
    """Compute SHAP values, saving ranked features and optional plot."""
    if shap is None:
        logger.info("shap not installed, skipping feature importance")
        return
    try:
        X_used = X_train
        if "scaler" in pipe.named_steps:
            X_used = pipe.named_steps["scaler"].transform(X_used)
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
        shap_values = explainer.shap_values(X_used)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        fi = pd.DataFrame(
            {
                "feature": features,
                "importance": np.abs(shap_values).mean(axis=0),
            }
        )
        out = LOG_DIR / "feature_importance.csv"
        fi.sort_values("importance", ascending=False).to_csv(out, index=False)
        logger.info("Logged feature importance to %s", out)
        if report_dir is not None:
            import matplotlib.pyplot as plt

            report_dir.mkdir(exist_ok=True)
            plt.figure()
            shap.summary_plot(shap_values, X_used, show=False, plot_type="bar")
            plt.tight_layout()
            plt.savefig(report_dir / "feature_importance.png")
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to compute SHAP values: %s", e)


@log_exceptions
def main(
    cfg: dict | None = None,
    export: bool = False,
    resume_online: bool = False,
    df_override: pd.DataFrame | None = None,
    transfer_from: str | None = None,
    use_pseudo_labels: bool = False,
) -> float:
    """Train LightGBM model and return weighted F1 score."""
    if cfg is None:
        cfg = load_config()
    if cfg.get("use_pseudo_labels"):
        use_pseudo_labels = True
    _subscribe_cpu_updates(cfg)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)
    scaler_path = root / "scaler.pkl"
    start_time = datetime.now()

    donor_booster = _load_donor_booster(transfer_from) if transfer_from else None

    mlflow.start_run("training", cfg)
    if df_override is not None:
        df = df_override
    else:
        symbols = cfg.get("symbols") or [cfg.get("symbol")]
        all_dfs = []
        chunk_size = cfg.get("stream_chunk_size", 100_000)
        stream = cfg.get("stream_history", False)
        for sym in symbols:
            if stream:
                # Stream history in configurable chunks to minimize memory usage
                pq_path = root / "data" / f"{sym}_history.parquet"
                if pq_path.exists():
                    for chunk in load_history_iter(pq_path, chunk_size):
                        chunk["Symbol"] = sym
                        all_dfs.append(chunk)
                else:
                    df_sym = load_history_config(
                        sym, cfg, root, validate=cfg.get("validate", False)
                    )
                    df_sym["Symbol"] = sym
                    all_dfs.append(df_sym)
            else:
                df_sym = load_history_config(
                    sym, cfg, root, validate=cfg.get("validate", False)
                )
                df_sym["Symbol"] = sym
                all_dfs.append(df_sym)

        df = pd.concat(all_dfs, ignore_index=True)
        save_history_parquet(df, root / "data" / "history.parquet")

        df = make_features(df, validate=cfg.get("validate", False))
        adapter_path = root / "domain_adapter.pkl"
        adapter = DomainAdapter.load(adapter_path)
        num_cols = df.select_dtypes(np.number).columns
        if len(num_cols) > 0:
            adapter.fit_source(df[num_cols])
            df[num_cols] = adapter.transform(df[num_cols])
            adapter.save(adapter_path)
        df = periodic_reclassification(df, step=cfg.get("regime_reclass_period", 500))
        if "Symbol" in df.columns:
            df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "news_sentiment",
        "market_regime",
    ]
    features.extend(
        [
            c
            for c in df.columns
            if c.startswith("cross_corr_")
            or c.startswith("factor_")
            or c.startswith("cross_mom_")
        ]
    )
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")

    if df_override is not None:
        mlflow.log_param("data_source", "override")
    else:
        mlflow.log_param("data_source", "config")

    df["tb_label"] = triple_barrier(
        df["mid"],
        cfg.get("pt_mult", 0.01),
        cfg.get("sl_mult", 0.01),
        cfg.get("max_horizon", 10),
    )
    y = df["tb_label"]
    features = select_features(df[features], y)
    feat_path = root / "selected_features.json"
    feat_path.write_text(json.dumps(features))

    X = df[features]

    if use_pseudo_labels:
        pseudo_dir = root / "data" / "pseudo_labels"
        if pseudo_dir.exists():
            files = list(pseudo_dir.glob("*.parquet")) + list(pseudo_dir.glob("*.csv"))
            for p in files:
                try:
                    if p.suffix == ".parquet":
                        df_pseudo = pd.read_parquet(p)
                    else:
                        df_pseudo = pd.read_csv(p)
                except Exception:
                    continue
                if "pseudo_label" not in df_pseudo.columns:
                    continue
                if not set(features).issubset(df_pseudo.columns):
                    continue
                X = pd.concat([X, df_pseudo[features]], ignore_index=True)
                y = pd.concat([y, df_pseudo["pseudo_label"]], ignore_index=True)

    if cfg.get("tune"):
        from tuning.tuning import tune_lightgbm

        def train_trial(params: dict, _trial) -> float:
            clf = LGBMClassifier(
                n_estimators=200,
                n_jobs=cfg.get("n_jobs") or monitor.capabilities.cpus,
                random_state=seed,
                **params,
            )
            tscv_inner = PurgedTimeSeriesSplit(
                n_splits=cfg.get("n_splits", 5),
                embargo=cfg.get("max_horizon", 0),
            )
            scores: list[float] = []
            for tr_idx, va_idx in tscv_inner.split(X):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                clf.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
                    verbose=False,
                )
                preds = clf.predict(X_va)
                rep = classification_report(y_va, preds, output_dict=True)
                scores.append(rep["weighted avg"]["f1-score"])
            return float(np.mean(scores))

        best_params = tune_lightgbm(train_trial, n_trials=cfg.get("n_trials", 20))
        cfg.update(best_params)

    if cfg.get("meta_train"):
        from analysis.meta_learning import meta_train_lgbm, save_meta_weights

        state = meta_train_lgbm(df, features)
        save_meta_weights(state, "lgbm")
        return 0.0

    if cfg.get("fine_tune"):
        from analysis.meta_learning import (
            _LinearModel,
            fine_tune_model,
            load_meta_weights,
            save_meta_weights,
        )
        from torch.utils.data import TensorDataset

        regime = int(df["market_regime"].iloc[-1])
        mask = df["market_regime"] == regime
        X_reg = torch.tensor(df.loc[mask, features].values, dtype=torch.float32)
        y_reg = torch.tensor(
            (df.loc[mask, "return"].shift(-1) > 0).astype(float).values[:-1],
            dtype=torch.float32,
        )
        dataset = TensorDataset(X_reg[:-1], y_reg)
        state = load_meta_weights("lgbm")
        new_state, _ = fine_tune_model(
            state, dataset, lambda: _LinearModel(len(features)), steps=5
        )
        save_meta_weights(new_state, "lgbm", regime=f"regime_{regime}")
        return 0.0

    if resume_online:
        batch_size = cfg.get("online_batch_size", 1000)
        ckpt = load_latest_checkpoint(cfg.get("checkpoint_dir"))
        if ckpt:
            start_batch = ckpt[0] + 1
            state = ckpt[1]
            pipe: Pipeline = state["model"]
            scaler_state = state.get("scaler_state")
            if scaler_state and "scaler" in pipe.named_steps:
                pipe.named_steps["scaler"].load_state_dict(scaler_state)
        else:
            start_batch = 0
            steps: list[tuple[str, object]] = []
            if cfg.get("use_scaler", True):
                steps.append(("scaler", FeatureScaler.load(scaler_path)))
            steps.append(
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=200,
                        n_jobs=cfg.get("n_jobs") or monitor.capabilities.cpus,
                        random_state=seed,
                        keep_training_booster=True,
                        **_lgbm_params(cfg),
                    ),
                )
            )
            pipe = Pipeline(steps)
        _register_clf(pipe.named_steps["clf"])
        n_batches = math.ceil(len(X) / batch_size)
        t_max = len(y) - 1
        half_life = cfg.get("time_decay_half_life")
        for batch_idx in range(start_batch, n_batches):
            start = batch_idx * batch_size
            end = min(len(X), start + batch_size)
            Xb = X.iloc[start:end]
            yb = y.iloc[start:end]
            if "scaler" in pipe.named_steps:
                scaler = pipe.named_steps["scaler"]
                if hasattr(scaler, "partial_fit"):
                    scaler.partial_fit(Xb)
                Xb = scaler.transform(Xb)
            clf = pipe.named_steps["clf"]
            init_model = (
                donor_booster
                if batch_idx == 0 and donor_booster is not None and not ckpt
                else (clf.booster_ if (batch_idx > 0 or ckpt) else None)
            )
            fit_kwargs = {"init_model": init_model}
            if half_life:
                decay = 0.5 ** ((t_max - np.arange(start, end)) / half_life)
                if cfg.get("balance_classes"):
                    cw = compute_sample_weight("balanced", yb)
                    sw = cw * decay
                else:
                    sw = decay
                fit_kwargs["sample_weight"] = sw
            elif cfg.get("balance_classes"):
                sw = compute_sample_weight("balanced", yb)
                fit_kwargs["sample_weight"] = sw
            clf.fit(Xb, yb, **fit_kwargs)
            state = {"model": pipe}
            if "scaler" in pipe.named_steps:
                state["scaler_state"] = pipe.named_steps["scaler"].state_dict()
            save_checkpoint(state, batch_idx, cfg.get("checkpoint_dir"))
        joblib.dump(pipe, root / "model.joblib")
        if "scaler" in pipe.named_steps:
            pipe.named_steps["scaler"].save(scaler_path)
        return 0.0

    tscv = PurgedTimeSeriesSplit(
        n_splits=cfg.get("n_splits", 5), embargo=cfg.get("max_horizon", 0)
    )
    all_preds: list[int] = []
    all_true: list[int] = []
    all_probs: list[float] = []
    all_lower: list[float] = []
    all_upper: list[float] = []
    final_pipe: Pipeline | None = None
    X_train_final: pd.DataFrame | None = None
    last_val_y: np.ndarray | None = None
    last_val_probs: np.ndarray | None = None
    start_fold = 0
    ckpt = load_latest_checkpoint(cfg.get("checkpoint_dir"))
    if ckpt:
        last_fold, state = ckpt
        start_fold = last_fold + 1
        all_preds = state.get("all_preds", [])
        all_true = state.get("all_true", [])
        all_probs = state.get("all_probs", [])
        all_lower = state.get("all_lower", [])
        all_upper = state.get("all_upper", [])
        final_pipe = state.get("model")
        scaler_state = state.get("scaler_state")
        if final_pipe and scaler_state and "scaler" in final_pipe.named_steps:
            final_pipe.named_steps["scaler"].load_state_dict(scaler_state)
        logger.info("Resuming from checkpoint at fold %s", last_fold)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if fold < start_fold:
            continue
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if cfg.get("use_data_augmentation", False) or cfg.get(
            "use_diffusion_aug", False
        ):
            fname = (
                "synthetic_sequences_diffusion.npz"
                if cfg.get("use_diffusion_aug", False)
                else "synthetic_sequences.npz"
            )
            aug_path = root / "data" / "augmented" / fname
            if aug_path.exists():
                data = np.load(aug_path)
                X_aug = data["X"][:, -1, :]
                y_aug = data["y"]
                df_aug = pd.DataFrame(X_aug, columns=features)
                X_train = pd.concat([X_train, df_aug], ignore_index=True)
                y_train = pd.concat(
                    [
                        y_train,
                        pd.Series(
                            y_aug, index=range(len(y_train), len(y_train) + len(y_aug))
                        ),
                    ],
                    ignore_index=True,
                )

        steps = []
        if cfg.get("use_scaler", True):
            steps.append(("scaler", FeatureScaler()))
        steps.append(
            (
                "clf",
                LGBMClassifier(
                    n_estimators=200,
                    n_jobs=cfg.get("n_jobs") or monitor.capabilities.cpus,
                    random_state=seed,
                    **_lgbm_params(cfg),
                ),
            )
        )
        pipe = Pipeline(steps)
        _register_clf(pipe.named_steps["clf"])

        fit_params = {"clf__eval_set": [(X_val, y_val)]}
        esr = cfg.get("early_stopping_rounds", 50)
        if esr:
            fit_params["clf__early_stopping_rounds"] = esr
        if donor_booster is not None:
            fit_params["clf__init_model"] = donor_booster
        half_life = cfg.get("time_decay_half_life")
        t_max = len(y) - 1
        if half_life:
            decay = 0.5 ** ((t_max - train_idx) / half_life)
            if cfg.get("balance_classes"):
                cw = compute_sample_weight("balanced", y_train)
                sw = cw * decay
            else:
                sw = decay
            fit_params["clf__sample_weight"] = sw
        elif cfg.get("balance_classes"):
            sw = compute_sample_weight("balanced", y_train)
            fit_params["clf__sample_weight"] = sw
        pipe.fit(X_train, y_train, **fit_params)

        probs = pipe.predict_proba(X_val)[:, 1]
        prec, rec, thr = precision_recall_curve(y_val, probs)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])
        best_thr = thr[best_idx]
        preds = (probs > best_thr).astype(int)
        mlflow.log_metric(f"fold_{fold}_best_thr", float(best_thr))
        last_val_y, last_val_probs = y_val, probs
        residuals = y_val.values - probs
        q = fit_residuals(residuals, alpha=cfg.get("interval_alpha", 0.1))
        lower, upper = predict_interval(probs, q)
        cov = evaluate_coverage(y_val, lower, upper)
        mlflow.log_metric(f"fold_{fold}_interval_coverage", cov)
        logger.info("Fold %d interval coverage: %.3f", fold, cov)
        conf = np.abs(probs - 0.5) * 2
        sizer = PositionSizer(capital=cfg.get("eval_capital", 1000.0))
        for p, c in zip(probs, conf):
            sizer.size(p, confidence=c)
        report = classification_report(y_val, preds, output_dict=True)
        logger.info("Fold %d\n%s", fold, classification_report(y_val, preds))
        mlflow.log_metric(
            f"fold_{fold}_f1_weighted", report["weighted avg"]["f1-score"]
        )

        all_preds.extend(preds)
        all_true.extend(y_val)
        all_probs.extend(probs)
        all_lower.extend(lower)
        all_upper.extend(upper)
        state = {
            "model": pipe,
            "all_preds": all_preds,
            "all_true": all_true,
            "all_probs": all_probs,
            "all_lower": all_lower,
            "all_upper": all_upper,
            "metrics": report,
        }
        if "scaler" in pipe.named_steps:
            state["scaler_state"] = pipe.named_steps["scaler"].state_dict()
        save_checkpoint(state, fold, cfg.get("checkpoint_dir"))

        if fold == tscv.n_splits - 1:
            final_pipe = pipe
            X_train_final = X_train

    overall_cov = (
        evaluate_coverage(all_true, all_lower, all_upper) if all_lower else 0.0
    )
    mlflow.log_metric("interval_coverage", overall_cov)
    logger.info("Overall interval coverage: %.3f", overall_cov)
    pred_df = pd.DataFrame(
        {
            "y_true": all_true,
            "pred": all_preds,
            "prob": all_probs,
            "lower": all_lower,
            "upper": all_upper,
        }
    )
    pred_path = root / "val_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    mlflow.log_artifact(str(pred_path))

    calibrator = None
    calib_method = cfg.get("calibration")
    if calib_method and final_pipe is not None:
        calibrator = ProbabilityCalibrator(method=calib_method).fit(
            last_val_y, last_val_probs
        )
        calibrated_probs = calibrator.predict(last_val_probs)
        log_reliability(
            last_val_y,
            last_val_probs,
            calibrated_probs,
            root / "reports" / "calibration",
            "lgbm",
            calib_method,
        )
        final_pipe = CalibratedModel(final_pipe, calibrator)

    aggregate_report = classification_report(all_true, all_preds, output_dict=True)
    logger.info("\n%s", classification_report(all_true, all_preds))

    joblib.dump(final_pipe, root / "model.joblib")
    if final_pipe and "scaler" in final_pipe.named_steps:
        final_pipe.named_steps["scaler"].save(scaler_path)
    logger.info("Model saved to %s", root / "model.joblib")
    mlflow.log_param("use_scaler", cfg.get("use_scaler", True))
    mlflow.log_metric("f1_weighted", aggregate_report["weighted avg"]["f1-score"])
    mlflow.log_artifact(str(root / "model.joblib"))
    version_id = model_store.save_model(
        final_pipe,
        cfg,
        {"f1_weighted": aggregate_report["weighted avg"]["f1-score"]},
        features=features,
    )
    logger.info("Registered model version %s", version_id)

    # Train dedicated models for each market regime
    base_features = [f for f in features if f != "market_regime"]
    regime_models: dict[int, Pipeline] = {}
    for regime in sorted(df["market_regime"].unique()):
        mask = df["market_regime"] == regime
        X_reg = df.loc[mask, base_features]
        y_reg = (df.loc[mask, "return"].shift(-1) > 0).astype(int)
        steps_reg: list[tuple[str, object]] = []
        if cfg.get("use_scaler", True):
            steps_reg.append(("scaler", FeatureScaler()))
        steps_reg.append(
            (
                "clf",
                LGBMClassifier(
                    n_estimators=200,
                    n_jobs=cfg.get("n_jobs") or monitor.capabilities.cpus,
                    random_state=seed,
                    **_lgbm_params(cfg),
                ),
            )
        )
        pipe_reg = Pipeline(steps_reg)
        _register_clf(pipe_reg.named_steps["clf"])
        pipe_reg.fit(X_reg, y_reg)
        regime_models[int(regime)] = pipe_reg
        logger.info("Trained regime-specific model for regime %s", regime)

    regimes_path = root / "regime_models.joblib"
    joblib.dump(regime_models, regimes_path)
    mlflow.log_artifact(str(regimes_path))
    logger.info("Regime-specific models saved to %s", regimes_path)

    out = root / "classification_report.json"
    with out.open("w") as f:
        json.dump(aggregate_report, f, indent=2)
    mlflow.log_artifact(str(out))

    if (
        final_pipe is not None
        and X_train_final is not None
        and cfg.get("feature_importance", False)
    ):
        report_dir = Path(__file__).resolve().parent / "reports"
        log_shap_importance(final_pipe, X_train_final, features, report_dir)

    if export and final_pipe is not None:
        from models.export import export_lightgbm

        sample = X.iloc[: min(len(X), 10)]
        if "scaler" in final_pipe.named_steps:
            sample = final_pipe.named_steps["scaler"].transform(sample)
        clf = final_pipe.named_steps.get("clf", final_pipe)
        export_lightgbm(clf, sample)

    # Active learning: queue uncertain samples and integrate new labels
    try:
        if final_pipe is not None:
            al_queue = ActiveLearningQueue()
            probs = final_pipe.predict_proba(X)
            al_queue.push(X.index, probs, k=cfg.get("al_queue_size", 10))
            new_labels = al_queue.pop_labeled()
            if not new_labels.empty and "tb_label" in df.columns:
                df = merge_labels(df, new_labels, "tb_label")
                save_history_parquet(df, root / "data" / "history.parquet")
    except Exception as e:  # pragma: no cover - fail safe
        logger.warning("Active learning step failed: %s", e)
    model_card.generate(
        cfg,
        [root / "data" / "history.parquet"],
        features,
        aggregate_report,
        root / "reports" / "model_cards",
    )
    mlflow.log_metric(
        "runtime",
        (datetime.now() - start_time).total_seconds(),
    )
    mlflow.end_run()
    return float(aggregate_report.get("weighted avg", {}).get("f1-score", 0.0))


def launch(
    cfg: dict | None = None,
    export: bool = False,
    resume_online: bool = False,
    transfer_from: str | None = None,
    use_pseudo_labels: bool = False,
) -> list[float]:
    """Launch training locally or across a Ray cluster."""
    if cfg is None:
        cfg = load_config()
    if cluster_available():
        seeds = cfg.get("seeds", [cfg.get("seed", 42)])
        results = []
        for s in seeds:
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            results.append(
                submit(
                    main,
                    cfg_s,
                    export=export,
                    resume_online=resume_online,
                    transfer_from=transfer_from,
                    use_pseudo_labels=use_pseudo_labels,
                )
            )
        return results
    return [
        main(
            cfg,
            export=export,
            resume_online=resume_online,
            transfer_from=transfer_from,
            use_pseudo_labels=use_pseudo_labels,
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument(
        "--evo-search",
        action="store_true",
        help="Run evolutionary multi-objective parameter search",
    )
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    parser.add_argument(
        "--resume-online",
        action="store_true",
        help="Resume incremental training from the latest checkpoint",
    )
    parser.add_argument(
        "--transfer-from",
        type=str,
        help="Initialize model using weights from a donor symbol",
    )
    parser.add_argument(
        "--meta-train",
        action="store_true",
        help="Run meta-training to produce meta-initialised weights",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune from meta weights on the latest regime",
    )
    parser.add_argument(
        "--use-pseudo-labels",
        action="store_true",
        help="Include pseudo-labeled samples during training",
    )
    args = parser.parse_args()
    ray_init()
    try:
        if args.tune:
            from tuning.bayesian_search import run_search

            cfg = load_config()

            def train_fn(c: dict, _trial) -> float:
                return main(c)

            run_search(train_fn, cfg)
        elif args.evo_search:
            from copy import deepcopy
            from tuning.evolutionary_search import run_evolutionary_search
            from backtest import run_backtest

            cfg = load_config()

            def eval_fn(params: dict) -> tuple[float, float, float]:
                trial_cfg = deepcopy(cfg)
                trial_cfg.update(params)
                main(trial_cfg)
                metrics = run_backtest(trial_cfg)
                return (
                    -float(metrics.get("return", 0.0)),
                    float(metrics.get("max_drawdown", 0.0)),
                    -float(metrics.get("trade_count", metrics.get("trades", 0.0))),
                )

            space = {
                "learning_rate": (1e-4, 2e-1, "log"),
                "num_leaves": (16, 255, "int"),
                "max_depth": (3, 12, "int"),
            }
            run_evolutionary_search(eval_fn, space)
        else:
            cfg = load_config()
            if args.meta_train:
                cfg["meta_train"] = True
            if args.fine_tune:
                cfg["fine_tune"] = True
            if args.use_pseudo_labels:
                cfg["use_pseudo_labels"] = True
            launch(
                cfg,
                export=args.export,
                resume_online=args.resume_online,
                transfer_from=args.transfer_from,
                use_pseudo_labels=args.use_pseudo_labels,
            )
    finally:
        ray_shutdown()
