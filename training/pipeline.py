"""Training routine for the Adaptive MT5 bot.

The training procedure uses :class:`analysis.purged_cv.PurgedTimeSeriesSplit`
with symbol identifiers as *groups* so that observations from the same
instrument never appear in both the training and validation folds.
"""

import os
from pathlib import Path
import random
import json
import logging
import asyncio
import joblib
import pandas as pd
import math
import weakref
from dataclasses import dataclass

try:
    import torch
except Exception:  # noqa: E722
    torch = None

if torch is not None:
    from torch.utils.data import DataLoader, TensorDataset
    from models.cross_modal_transformer import CrossModalTransformer
    from models.cross_modal_classifier import CrossModalClassifier
else:  # pragma: no cover - torch may be unavailable in some environments
    TensorDataset = None  # type: ignore
    CrossModalTransformer = None  # type: ignore
    CrossModalClassifier = None  # type: ignore
from analysis.purged_cv import PurgedTimeSeriesSplit
from analysis.data_quality import score_samples as dq_score_samples
from lightgbm import LGBMClassifier
from analytics import mlflow_client as mlflow
from datetime import datetime

try:
    from data.feature_scaler import FeatureScaler
except Exception:  # pragma: no cover - optional dependency
    class FeatureScaler:  # type: ignore[override]
        """Minimal scaler stub used when optional dependency is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self, X, y=None):  # type: ignore[override]
            return self

        def transform(self, X):  # type: ignore[override]
            return X

        def fit_transform(self, X, y=None):  # type: ignore[override]
            return X

        @classmethod
        def load(cls, path):  # type: ignore[override]
            return cls()

        def save(self, path):  # pragma: no cover - compatibility shim
            return path
from mt5.train_utils import prepare_modal_arrays
from mt5.log_utils import setup_logging, log_exceptions, LOG_DIR
import numpy as np
from risk.position_sizer import PositionSizer
from mt5.ray_utils import (
    init as ray_init,
    shutdown as ray_shutdown,
    cluster_available,
    submit,
)

try:
    import shap
except Exception:  # noqa: E722
    shap = None

from utils import ensure_environment, load_config
from features import start_capability_watch
from mt5.config_models import AppConfig
from utils.resource_monitor import monitor
from mt5.state_manager import save_checkpoint, load_latest_checkpoint
from models import model_store
from analysis.prob_calibration import (
    ProbabilityCalibrator,
    CalibratedModel,
    log_reliability,
)
from analysis.active_learning import ActiveLearningQueue, merge_labels
from analysis import model_card
from reports.run_history import RunHistoryRecorder
from analysis.domain_adapter import DomainAdapter
from data.history import save_history_parquet
from models.conformal import (
    ConformalIntervalParams,
    calibrate_intervals,
    evaluate_coverage,
    fit_residuals,
)
from analysis.regime_thresholds import find_regime_thresholds
from analysis.concept_drift import ConceptDriftMonitor
from analysis.pseudo_labeler import generate_pseudo_labels
from analysis.risk_loss import cvar, max_drawdown, risk_penalty, RiskBudget
from analysis.multi_objective import (
    TradeMetrics,
    compute_metrics as mo_compute_metrics,
    weighted_sum as mo_weighted_sum,
)
from models.meta_label import train_meta_classifier
from analysis.evaluate import (
    bootstrap_classification_metrics,
    risk_adjusted_metrics,
)
from analysis.interpret_model import generate_shap_report
from analysis.similar_days import add_similar_day_features
from training.curriculum import CurriculumScheduler
from models.multi_task_heads import MultiTaskHeadEstimator

from training.data_loader import StreamingTrainingFrame, load_training_frame
from training.features import (
    apply_domain_adaptation,
    append_risk_profile_features,
    build_feature_candidates,
    select_model_features,
)
from training.preprocessing import FeatureSanitizer
from training.labels import generate_training_labels
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from training.postprocess import (
    summarise_predictions,
    build_model_metadata,
    persist_model,
)
from training.utils import combined_sample_weight

logger = logging.getLogger(__name__)


@dataclass
class HistoryLoadResult:
    """Data returned from :func:`load_histories`."""

    frame: pd.DataFrame | StreamingTrainingFrame
    data_source: str
    stream_metadata: dict[str, object] | None
    stream: bool
    chunk_size: int
    feature_lookback: int
    validate: bool


@dataclass
class FeaturePreparationResult:
    """Feature engineering output used to build datasets."""

    df: pd.DataFrame
    features: list[str]
    labels: pd.DataFrame
    label_cols: list[str]
    abs_label_cols: list[str]
    vol_label_cols: list[str]
    sel_target: pd.Series | None
    use_multi_task_heads: bool
    user_budget: RiskBudget | None


@dataclass
class DatasetBuildResult:
    """Datasets ready for model training."""

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.DataFrame
    features: list[str]
    label_cols: list[str]
    abs_label_cols: list[str]
    vol_label_cols: list[str]
    groups: pd.Series | pd.Index | None
    timestamps: np.ndarray
    al_queue: ActiveLearningQueue
    al_threshold: float
    queue_stats: dict[str, int]
    risk_budget: RiskBudget | None
    user_budget: RiskBudget | None


@dataclass
class TrainingResult:
    """Outcome of the model training stage."""

    final_pipe: Pipeline | None
    features: list[str]
    aggregate_report: dict[str, object]
    boot_metrics: dict[str, object]
    base_f1: float
    regime_thresholds: dict[int, float]
    overall_params: ConformalIntervalParams | None
    overall_q: dict[int, float]
    overall_cov: float
    all_probs: list[float]
    all_conf: list[float]
    all_true: list[int]
    al_queue: ActiveLearningQueue
    al_threshold: float
    df: pd.DataFrame
    X: pd.DataFrame
    X_train_final: pd.DataFrame | None
    risk_budget: RiskBudget | None
    model_metadata: dict[str, object]
    f1_ci: tuple[float, float]
    prec_ci: tuple[float, float]
    rec_ci: tuple[float, float]
    final_score: float
    should_log_artifacts: bool


@dataclass
class ArtifactLogResult:
    """Summary from :func:`log_artifacts`."""

    meta_model_id: str | None
    model_version_id: str | None
    pseudo_label_path: Path | None
    queued_count: int


def init_logging() -> logging.Logger:
    """Initialise structured logging for the training pipeline."""

    setup_logging()
    return logging.getLogger(__name__)

# Track active classifiers for dynamic resizing
_ACTIVE_CLFS: weakref.WeakSet[LGBMClassifier] = weakref.WeakSet()


def _register_clf(clf: LGBMClassifier) -> None:
    """Keep reference to classifier for dynamic n_jobs updates."""
    _ACTIVE_CLFS.add(clf)


def _prune_finished_classifiers() -> None:
    """Drop any classifiers that are no longer strongly referenced."""

    global _ACTIVE_CLFS
    if not _ACTIVE_CLFS:
        return
    _ACTIVE_CLFS = weakref.WeakSet(_ACTIVE_CLFS)


def _normalise_regime_thresholds(
    thresholds: dict[int | str, float | int] | None,
) -> dict[int, float]:
    """Return thresholds with integer keys and float values."""

    if not thresholds:
        return {}
    return {int(k): float(v) for k, v in thresholds.items()}


def _lgbm_params(cfg: AppConfig) -> dict:
    """Extract LightGBM hyper-parameters from config."""
    params: dict[str, float | int] = {}
    if cfg.training.num_leaves is not None:
        params["num_leaves"] = cfg.training.num_leaves
    if cfg.training.learning_rate is not None:
        params["learning_rate"] = cfg.training.learning_rate
    if cfg.training.max_depth is not None:
        params["max_depth"] = cfg.training.max_depth
    return params


def _cfg_get(cfg: AppConfig | dict | None, key: str, default: object = None) -> object:
    """Best-effort lookup supporting nested ``training`` sections."""

    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        value = getter(key, None)
        if value is not None:
            return value
    value = getattr(cfg, key, None)
    if value is not None:
        return value
    training = getattr(cfg, "training", None)
    if training is None:
        return default
    if isinstance(training, dict):
        return training.get(key, default)
    training_get = getattr(training, "get", None)
    if callable(training_get):
        value = training_get(key, None)
        if value is not None:
            return value
    value = getattr(training, key, None)
    return value if value is not None else default


def _make_sanitizer(cfg: AppConfig | dict | None) -> FeatureSanitizer:
    """Instantiate :class:`FeatureSanitizer` respecting configuration."""

    method = _cfg_get(cfg, "sanitizer_fill_method", None)
    if method is None:
        method = _cfg_get(cfg, "sanitizer_fill", "median")
    value = _cfg_get(cfg, "sanitizer_fill_value", None)
    try:
        method = str(method).lower()
    except Exception:
        method = "median"
    return FeatureSanitizer(fill_method=method, fill_value=value)


def _subscribe_cpu_updates(cfg: AppConfig) -> None:
    async def _watch() -> None:
        q = monitor.subscribe()
        while True:
            await q.get()
            n_jobs = cfg.get("n_jobs") or monitor.capabilities.cpus
            stale_seen = False
            for c in tuple(_ACTIVE_CLFS):
                if c is None:
                    stale_seen = True
                    continue
                try:
                    c.set_params(n_jobs=n_jobs)
                except ReferenceError:
                    stale_seen = True
                except Exception:
                    logger.debug("Failed to update n_jobs for classifier")
            if stale_seen:
                _prune_finished_classifiers()

    monitor.create_task(_watch())


def _index_to_timestamps(idx: pd.Index) -> np.ndarray:
    """Convert an index or Series to a numeric timestamp array."""
    if isinstance(idx, pd.Series):
        idx = idx.index if idx.name is None else idx
    if isinstance(idx, pd.DatetimeIndex):
        return idx.view("int64")
    arr = np.asarray(idx)
    try:
        return arr.astype("int64")
    except Exception:  # pragma: no cover - best effort conversion
        return np.arange(len(idx), dtype="int64")


def _train_cross_modal_feature(
    price: np.ndarray,
    news: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 64,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
) -> np.ndarray:
    """Train a small cross-modal transformer and return fused probabilities."""

    if torch is None or CrossModalTransformer is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for cross-modal fusion")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(
        torch.tensor(price, dtype=torch.float32),
        torch.tensor(news, dtype=torch.float32),
        torch.tensor(labels.astype(np.float32), dtype=torch.float32),
    )
    batch = max(1, min(batch_size, len(dataset)))
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    model = CrossModalTransformer(
        price_dim=price.shape[-1],
        news_dim=news.shape[-1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    for _ in range(max(1, epochs)):
        model.train()
        for price_batch, news_batch, target in loader:
            price_batch = price_batch.to(device)
            news_batch = news_batch.to(device)
            target = target.to(device)
            opt.zero_grad()
            preds = model(price_batch, news_batch)
            loss = loss_fn(preds, target)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(
            torch.tensor(price, dtype=torch.float32, device=device),
            torch.tensor(news, dtype=torch.float32, device=device),
        ).cpu().numpy()
    return preds


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


def _maybe_generate_indicators(
    X: pd.DataFrame,
    hypernet: "torch.nn.Module | None" = None,
    asset_features: np.ndarray | None = None,
    regime: np.ndarray | None = None,
    registry_path: Path | None = None,
    evolved_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int] | None]:
    """Augment ``X`` with persisted and newly generated indicators."""

    from features import auto_indicators

    # Always append any previously evolved indicators and persisted auto
    # indicators.  ``auto_indicators.apply`` loads all versioned formula files
    # from ``formula_dir``.
    formula_dir: Path | None = None
    if evolved_path is not None:
        formula_dir = evolved_path if evolved_path.is_dir() else evolved_path.parent
    X_aug = auto_indicators.apply(
        X,
        registry_path=registry_path or auto_indicators.REGISTRY_PATH,
        formula_dir=formula_dir,
    )

    if hypernet is None:
        return X_aug, None

    asset = (
        asset_features
        if asset_features is not None
        else np.mean(X_aug.values, axis=0, keepdims=True)
    )
    reg = regime if regime is not None else np.zeros((1, 1))
    X_new, desc = auto_indicators.generate(
        X_aug,
        hypernet,
        asset,
        reg,
        registry_path=registry_path or auto_indicators.REGISTRY_PATH,
    )
    return X_new, desc


def _maybe_evolve_on_degradation(
    metric: float,
    baseline: float | None,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    threshold: float = 0.05,
    path: Path | None = None,
) -> bool:
    """Trigger indicator evolution when performance drops.

    The function writes candidate indicator formulas to the feature store with a
    monotonically increasing version tag.  Before persisting the formulas it
    compares their ``score`` against the current validation ``metric``.  Only if
    the best evolved indicator outperforms the baseline metric will the formulas
    remain in the store.
    """

    if baseline is None or baseline - metric < threshold:
        return False

    from analysis import indicator_evolution as ind_evo
    from datetime import datetime
    import json

    base_dir = (
        path if path is not None else Path(__file__).resolve().parent / "feature_store"
    )
    if base_dir.suffix:
        # Caller provided an explicit file path; keep as-is
        store_path = base_dir
        base_dir = base_dir.parent
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(base_dir.glob("evolved_indicators_v*.json"))
        store_path = base_dir / f"evolved_indicators_v{len(existing) + 1}.json"

    inds = ind_evo.evolve(X, y, store_path)
    if not inds:
        store_path.unlink(missing_ok=True)
        return False

    best = inds[0]
    if best.score <= metric:
        logger.info(
            "Discarding evolved indicator %s score %.4f <= baseline %.4f",
            best.name,
            best.score,
            metric,
        )
        store_path.unlink(missing_ok=True)
        return False

    # Document provenance of the promoted indicators
    meta_path = store_path.with_name(f"{store_path.stem}.meta.json")
    meta = {
        "baseline_metric": baseline,
        "current_metric": metric,
        "best_score": best.score,
        "created": datetime.utcnow().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    for ind in inds:
        logger.info("Evolved indicator %s score %.4f", ind.name, ind.score)
    return True


def train_multi_output_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: dict | None = None,
    hypernet: "torch.nn.Module | None" = None,
    asset_features: np.ndarray | None = None,
    regime_tag: np.ndarray | None = None,
    registry_path: Path | None = None,
    *,
    baseline_metric: float | None = None,
    degradation: float = 0.05,
    evolve_path: Path | None = None,
) -> tuple[Pipeline, dict[str, object]]:
    """Train a shared-trunk multi-task model and return per-task metrics."""

    cfg = cfg or {}
    steps: list[tuple[str, object]] = [("sanitizer", _make_sanitizer(cfg))]
    if cfg.get("use_scaler", True):
        steps.append(("scaler", FeatureScaler()))

    X, ind_desc = _maybe_generate_indicators(
        X,
        hypernet,
        asset_features=asset_features,
        regime=regime_tag,
        registry_path=registry_path,
        evolved_path=evolve_path,
    )

    label_cols = [c for c in y.columns if c.startswith("direction_")]
    abs_cols = [c for c in y.columns if c.startswith("abs_return_")]
    vol_cols = [c for c in y.columns if c.startswith("volatility_")]

    hidden_default = 32
    if isinstance(X, pd.DataFrame):
        hidden_default = max(16, len(X.columns) // 2 or 16)

    head_params = {
        "classification_targets": label_cols,
        "abs_targets": abs_cols,
        "volatility_targets": vol_cols,
        "hidden_dim": int(cfg.get("head_hidden_dim", hidden_default)),
        "learning_rate": float(cfg.get("head_learning_rate", 0.01)),
        "epochs": int(cfg.get("head_epochs", 200)),
        "classification_weight": float(cfg.get("head_classification_weight", 1.0)),
        "abs_weight": float(cfg.get("head_abs_weight", 1.0)),
        "volatility_weight": float(cfg.get("head_volatility_weight", 1.0)),
        "l2": float(cfg.get("head_l2", 0.0)),
        "random_state": cfg.get("seed"),
    }

    head = MultiTaskHeadEstimator(**head_params)
    steps.append(("multi_task", head))
    pipe = Pipeline(steps)
    pipe.classification_targets_ = list(label_cols)
    pipe.abs_return_targets_ = list(abs_cols)
    pipe.volatility_targets_ = list(vol_cols)
    pipe.regression_feature_columns_ = (
        list(X.columns) if isinstance(X, pd.DataFrame) else None
    )
    pipe.head_config_ = {}
    pipe.classification_thresholds_ = {}

    pipe.fit(X, y)
    sanitizer = pipe.named_steps.get("sanitizer")
    if sanitizer is not None and hasattr(sanitizer, "state_dict"):
        pipe.sanitizer_state_ = sanitizer.state_dict()

    reports: dict[str, object] = {}
    f1_scores: list[float] = []
    sharpe_scores: list[float] = []
    calmar_scores: list[float] = []
    threshold_metric = cfg.get("threshold_metric", "f1")
    est: MultiTaskHeadEstimator = pipe.named_steps["multi_task"]
    head_config = dict(est.head_config_)
    pipe.head_config_ = head_config
    pipe.primary_label_ = est.primary_label_

    if len(pipe.steps) > 1:
        X_shared = pipe[:-1].transform(X)
    else:
        X_shared = X

    preds = (
        np.zeros((len(y), len(label_cols)), dtype=int)
        if label_cols
        else np.zeros((len(y), 0), dtype=int)
    )
    thresholds: dict[str, float] = {}
    if label_cols:
        class_probs = est.predict_classification_proba(X_shared)
        class_probs = np.asarray(class_probs, dtype=float)
        if class_probs.ndim == 1:
            class_probs = class_probs.reshape(-1, 1)
        for i, col in enumerate(label_cols):
            probs = class_probs[:, i]
            if threshold_metric == "f1":
                precision, recall, thresholds_arr = precision_recall_curve(y[col], probs)
                f1 = 2 * precision * recall / (precision + recall + 1e-12)
                if len(thresholds_arr) > 0:
                    best_idx = int(np.argmax(f1[:-1]))
                    best_thr = float(thresholds_arr[best_idx])
                    best_metric = float(f1[best_idx])
                else:
                    best_thr = 0.5
                    best_metric = 0.0
            else:
                unique_thr = np.unique(probs)
                best_thr = 0.5
                best_metric = -np.inf
                for thr in unique_thr:
                    pred = (probs >= thr).astype(int)
                    metrics = risk_adjusted_metrics(y[col], pred)
                    metric_val = metrics.get(threshold_metric, float("-inf"))
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_thr = float(thr)
            thresholds[col] = best_thr
            preds[:, i] = (probs >= best_thr).astype(int)
            rep = classification_report(y[col], preds[:, i], output_dict=True)
            risk = risk_adjusted_metrics(y[col], preds[:, i])
            reports[col] = {**rep, **risk}
            f1_val = rep["weighted avg"]["f1-score"]
            precision_val = rep["weighted avg"]["precision"]
            recall_val = rep["weighted avg"]["recall"]
            f1_scores.append(f1_val)
            sharpe_scores.append(risk["sharpe"])
            calmar_scores.append(risk["calmar"])
            logger.info(
                "Best threshold for %s (%s): %.4f", col, threshold_metric, best_thr
            )
            logger.info(
                "Validation metrics %s | F1 %.4f | Precision %.4f | Recall %.4f | Sharpe %.4f | Calmar %.4f",
                col,
                f1_val,
                precision_val,
                recall_val,
                risk["sharpe"],
                risk["calmar"],
            )
            try:
                mlflow.log_metric(f"thr_{col}", best_thr)
                mlflow.log_metric(f"{threshold_metric}_{col}", best_metric)
                mlflow.log_metric(f"sharpe_{col}", risk["sharpe"])
                mlflow.log_metric(f"calmar_{col}", risk["calmar"])
                mlflow.log_metric(f"f1_{col}", f1_val)
                mlflow.log_metric(f"precision_{col}", precision_val)
                mlflow.log_metric(f"recall_{col}", recall_val)
            except Exception:  # pragma: no cover - mlflow optional
                pass
        est.thresholds_ = thresholds
        if label_cols:
            est.classes_ = np.array([0, 1], dtype=int)
        pipe.classification_thresholds_ = thresholds
        head_config["thresholds"] = thresholds

    if label_cols and f1_scores:
        reports["aggregate_f1"] = float(np.mean(f1_scores))
        try:  # pragma: no cover - mlflow optional
            mlflow.log_metric("aggregate_f1", reports["aggregate_f1"])
        except Exception:
            pass
    if sharpe_scores:
        reports["aggregate_sharpe"] = float(np.mean(sharpe_scores))
        try:  # pragma: no cover - mlflow optional
            mlflow.log_metric("aggregate_sharpe", reports["aggregate_sharpe"])
        except Exception:
            pass
    if calmar_scores:
        reports["aggregate_calmar"] = float(np.mean(calmar_scores))
        try:  # pragma: no cover - mlflow optional
            mlflow.log_metric("aggregate_calmar", reports["aggregate_calmar"])
        except Exception:
            pass

    rmse_scores: list[float] = []
    abs_rmse: list[float] = []
    vol_rmse: list[float] = []
    regression_heads: dict[str, dict[str, object]] = {}
    if abs_cols:
        abs_pred = est.predict_regression(X_shared, "abs_return")
        abs_pred = np.asarray(abs_pred, dtype=float)
        if abs_pred.ndim == 1:
            abs_pred = abs_pred.reshape(-1, 1)
        for i, col in enumerate(abs_cols):
            target = y[col].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((abs_pred[:, i] - target) ** 2)))
            preds = abs_pred[:, i]
            pred_mean = float(np.mean(preds))
            realized_mean = float(np.mean(target))
            bias = float(np.mean(preds - target))
            corr = float("nan")
            std_pred = float(np.std(preds))
            std_true = float(np.std(target))
            if std_pred > 1e-12 and std_true > 1e-12:
                corr = float(np.corrcoef(preds, target)[0, 1])
            reports[col] = {
                "rmse": rmse,
                "pred_mean": pred_mean,
                "realized_mean": realized_mean,
                "bias": bias,
                "corr": corr,
            }
            rmse_scores.append(rmse)
            abs_rmse.append(rmse)
            logger.info("RMSE for %s: %.4f", col, rmse)
            try:  # pragma: no cover - mlflow optional
                mlflow.log_metric(f"rmse_{col}", rmse)
                mlflow.log_metric(f"pred_mean_{col}", pred_mean)
                mlflow.log_metric(f"realized_mean_{col}", realized_mean)
                mlflow.log_metric(f"bias_{col}", bias)
                if not math.isnan(corr):
                    mlflow.log_metric(f"corr_{col}", corr)
            except Exception:
                pass
        regression_heads["abs_return"] = {"type": "multi_task", "columns": list(abs_cols)}
    if vol_cols:
        vol_pred = est.predict_regression(X_shared, "volatility")
        vol_pred = np.asarray(vol_pred, dtype=float)
        if vol_pred.ndim == 1:
            vol_pred = vol_pred.reshape(-1, 1)
        for i, col in enumerate(vol_cols):
            target = y[col].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((vol_pred[:, i] - target) ** 2)))
            preds = vol_pred[:, i]
            pred_mean = float(np.mean(preds))
            realized_mean = float(np.mean(target))
            bias = float(np.mean(preds - target))
            corr = float("nan")
            std_pred = float(np.std(preds))
            std_true = float(np.std(target))
            if std_pred > 1e-12 and std_true > 1e-12:
                corr = float(np.corrcoef(preds, target)[0, 1])
            reports[col] = {
                "rmse": rmse,
                "pred_mean": pred_mean,
                "realized_mean": realized_mean,
                "bias": bias,
                "corr": corr,
            }
            rmse_scores.append(rmse)
            vol_rmse.append(rmse)
            logger.info("RMSE for %s: %.4f", col, rmse)
            try:  # pragma: no cover - mlflow optional
                mlflow.log_metric(f"rmse_{col}", rmse)
                mlflow.log_metric(f"pred_mean_{col}", pred_mean)
                mlflow.log_metric(f"realized_mean_{col}", realized_mean)
                mlflow.log_metric(f"bias_{col}", bias)
                if not math.isnan(corr):
                    mlflow.log_metric(f"corr_{col}", corr)
            except Exception:
                pass
        regression_heads["volatility"] = {"type": "multi_task", "columns": list(vol_cols)}

    pipe.regression_heads_ = regression_heads
    pipe.regression_target_columns_ = {
        name: info["columns"] for name, info in regression_heads.items()
    }

    if rmse_scores:
        reports["aggregate_rmse"] = float(np.mean(rmse_scores))
        try:  # pragma: no cover - mlflow optional
            mlflow.log_metric("aggregate_rmse", reports["aggregate_rmse"])
        except Exception:
            pass
    if abs_rmse:
        reports["aggregate_abs_return_rmse"] = float(np.mean(abs_rmse))
        try:  # pragma: no cover - mlflow optional
            mlflow.log_metric(
                "aggregate_abs_return_rmse", reports["aggregate_abs_return_rmse"]
            )
        except Exception:
            pass
    if vol_rmse:
        reports["aggregate_volatility_rmse"] = float(np.mean(vol_rmse))
        try:  # pragma: no cover - mlflow optional
            mlflow.log_metric(
                "aggregate_volatility_rmse", reports["aggregate_volatility_rmse"]
            )
        except Exception:
            pass



def make_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Create focal loss objective for LightGBM.

    Parameters
    ----------
    alpha: float
        Weighting factor for the rare class.
    gamma: float
        Focusing parameter for modulating factor.

    Returns
    -------
    Callable
        Function computing gradient and hessian for LightGBM.
    """

    def _focal_loss(y_pred: np.ndarray, data) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(data, np.ndarray):
            y_true, y_pred = y_pred, data
        elif hasattr(data, "get_label"):
            y_true = data.get_label()
        else:
            y_true = data
        p = 1.0 / (1.0 + np.exp(-y_pred))
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        p_t = p * y_true + (1 - p) * (1 - y_true)
        mod_factor = (1 - p_t) ** gamma
        grad = (p - y_true) * alpha_t * mod_factor
        hess = alpha_t * mod_factor * p * (1 - p)
        return grad, hess

    return _focal_loss


def make_focal_loss_metric(alpha: float = 0.25, gamma: float = 2.0):
    """Create focal loss evaluation metric for LightGBM."""

    def _focal_metric(y_pred: np.ndarray, data) -> tuple[str, float, bool]:
        if isinstance(data, np.ndarray):
            y_true, y_pred = y_pred, data
        elif hasattr(data, "get_label"):
            y_true = data.get_label()
        else:
            y_true = data
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = p * y_true + (1 - p) * (1 - y_true)
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = -alpha_t * (1 - p_t) ** gamma * np.log(np.clip(p_t, 1e-7, 1 - 1e-7))
        return "focal_loss", float(np.mean(loss)), False

    return _focal_metric


def load_histories(
    cfg: AppConfig,
    root: Path,
    df_override: pd.DataFrame | None = None,
) -> HistoryLoadResult:
    """Load historical data and apply domain adaptation."""

    validate_flag = bool(cfg.get("validate", False))
    chunk_size_raw = cfg.get("stream_chunk_size", 100_000)
    chunk_size = int(chunk_size_raw if chunk_size_raw is not None else 100_000)
    stream = bool(cfg.get("stream_history", False))
    feature_lookback_raw = cfg.get("stream_feature_lookback", 512)
    feature_lookback = int(feature_lookback_raw if feature_lookback_raw is not None else 512)
    training_data, data_source = load_training_frame(
        cfg,
        root,
        df_override=df_override,
        stream=stream,
        chunk_size=chunk_size,
        feature_lookback=feature_lookback,
        validate=validate_flag,
    )
    stream_metadata: dict[str, object] | None = None
    if isinstance(training_data, StreamingTrainingFrame):
        stream_metadata = training_data.metadata
    adapter_path = root / "domain_adapter.pkl"
    df = apply_domain_adaptation(
        training_data,
        adapter_path,
        regime_step=cfg.get("regime_reclass_period", 500),
    )
    return HistoryLoadResult(
        frame=df,
        data_source=str(data_source),
        stream_metadata=stream_metadata,
        stream=stream,
        chunk_size=chunk_size,
        feature_lookback=feature_lookback,
        validate=validate_flag,
    )


def prepare_features(
    history: HistoryLoadResult,
    cfg: AppConfig,
    model_type: str,
    root: Path,
) -> FeaturePreparationResult:
    """Prepare features and labels for downstream stages."""

    df = history.frame
    rp = cfg.strategy.risk_profile
    user_budget = append_risk_profile_features(df, rp)
    features = build_feature_candidates(df, user_budget, cfg=cfg)
    if isinstance(df, StreamingTrainingFrame):
        column_source = df.collect_columns()
    else:
        column_source = list(df.columns)

    price_window_cols = sorted(c for c in column_source if c.startswith("price_window_"))
    news_emb_cols = sorted(c for c in column_source if c.startswith("news_emb_"))
    if model_type == "cross_modal":
        if not price_window_cols or not news_emb_cols:
            raise ValueError(
                "model_type='cross_modal' requires price_window_ and news_emb_ columns"
            )
        for col in price_window_cols + news_emb_cols:
            if col not in features:
                features.append(col)

    horizons = cfg.get("horizons", [cfg.get("max_horizon", 10)])
    labels_frame = generate_training_labels(
        df,
        stream=history.stream,
        horizons=horizons,
        chunk_size=history.chunk_size,
    )

    if isinstance(df, StreamingTrainingFrame):
        column_source = df.collect_columns()
        label_cols = [c for c in column_source if c.startswith("direction_")]
        abs_label_cols = [c for c in column_source if c.startswith("abs_return_")]
        vol_label_cols = [c for c in column_source if c.startswith("volatility_")]
        df_materialised = df.materialise()
        if label_cols or abs_label_cols or vol_label_cols:
            label_subset = label_cols + abs_label_cols + vol_label_cols
            labels_frame = df_materialised.loc[:, label_subset]
        else:
            labels_frame = pd.DataFrame(index=df_materialised.index)
        df = df_materialised
    else:
        df = pd.concat([df, labels_frame], axis=1)
        label_cols = [c for c in labels_frame.columns if c.startswith("direction_")]
        abs_label_cols = [c for c in labels_frame.columns if c.startswith("abs_return_")]
        vol_label_cols = [c for c in labels_frame.columns if c.startswith("volatility_")]

    use_multi_task_heads = bool(cfg.get("use_multi_task_heads", True))
    if model_type == "neural":
        use_multi_task_heads = True
    if model_type == "cross_modal":
        use_multi_task_heads = False
    elif not abs_label_cols and not vol_label_cols:
        use_multi_task_heads = bool(cfg.get("use_multi_task_heads", False))

    sel_target: pd.Series | None = None
    if label_cols:
        sel_target = labels_frame[label_cols[0]]
    elif not labels_frame.empty:
        sel_target = labels_frame.iloc[:, 0]

    mandatory_cols = [
        col
        for col in ["risk_tolerance", "leverage_cap", "drawdown_limit"]
        if col in df.columns
    ]
    features = select_model_features(
        df,
        features,
        sel_target,
        model_type=model_type,
        mandatory=mandatory_cols,
    )

    cross_modal_cfg = cfg.get("cross_modal_feature", {})
    if sel_target is not None and torch is not None and CrossModalTransformer is not None:
        try:
            modal_result = prepare_modal_arrays(
                df, sel_target.to_numpy(dtype=np.float32)
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            modal_result = None
            logger.warning("Failed to prepare modal arrays: %s", exc)
        else:
            if modal_result is not None:
                (
                    price_modal,
                    news_modal,
                    labels_modal,
                    mask_modal,
                    price_cols,
                    news_cols,
                ) = modal_result
                cfg.setdefault("model", {}).setdefault(
                    "cross_modal_features",
                    {"price": price_cols, "news": news_cols},
                )
                if (
                    model_type != "cross_modal"
                    and torch is not None
                    and CrossModalTransformer is not None
                ):
                    min_samples = int(cross_modal_cfg.get("min_samples", 50))
                    if labels_modal is not None and len(price_modal) >= max(10, min_samples):
                        try:
                            preds = _train_cross_modal_feature(
                                price_modal,
                                news_modal,
                                labels_modal,
                                epochs=int(cross_modal_cfg.get("epochs", 5)),
                                batch_size=int(cross_modal_cfg.get("batch_size", 128)),
                                d_model=int(cross_modal_cfg.get("d_model", 64)),
                                nhead=int(cross_modal_cfg.get("nhead", 4)),
                                num_layers=int(cross_modal_cfg.get("num_layers", 2)),
                                dropout=float(cross_modal_cfg.get("dropout", 0.1)),
                                lr=float(cross_modal_cfg.get("lr", 1e-3)),
                            )
                        except Exception as exc:  # pragma: no cover - fallback on failure
                            logger.warning("Cross-modal fusion failed: %s", exc)
                        else:
                            fused = np.full(len(df), 0.5, dtype=float)
                            fused[mask_modal] = preds
                            df["cross_modal_signal"] = fused
                            if "cross_modal_signal" not in features:
                                features.append("cross_modal_signal")
                            logger.info(
                                "Cross-modal feature trained on %d samples", len(price_modal)
                            )

    feat_path = root / "selected_features.json"
    feat_path.write_text(json.dumps(features))
    index_path = root / "similar_days_index.pkl"
    df, _ = add_similar_day_features(
        df,
        feature_cols=features,
        return_col="return",
        k=cfg.get("nn_k", 5),
        index_path=index_path,
    )
    for derived in ["nn_return_mean", "nn_vol"]:
        if derived not in features:
            features.append(derived)

    return FeaturePreparationResult(
        df=df,
        features=features,
        labels=labels_frame,
        label_cols=list(label_cols),
        abs_label_cols=list(abs_label_cols),
        vol_label_cols=list(vol_label_cols),
        sel_target=sel_target,
        use_multi_task_heads=use_multi_task_heads,
        user_budget=user_budget,
    )


def build_datasets(
    features: FeaturePreparationResult,
    cfg: AppConfig,
    root: Path,
    use_pseudo_labels: bool,
    risk_target: dict | None,
) -> DatasetBuildResult:
    """Construct model-ready datasets from engineered features."""

    df = features.df.copy()
    feature_cols = list(features.features)
    label_cols = list(features.label_cols)
    abs_label_cols = list(features.abs_label_cols)
    vol_label_cols = list(features.vol_label_cols)
    y = features.labels.copy()

    risk_budget = None
    if risk_target:
        risk_budget = RiskBudget(
            max_leverage=float(risk_target.get("max_leverage", 1.0)),
            max_drawdown=float(risk_target.get("max_drawdown", 0.0)),
            cvar_limit=risk_target.get("cvar"),
        )
        for name, val in risk_budget.as_features().items():
            df[name] = val
            if name not in feature_cols:
                feature_cols.append(name)
        if "position" in df.columns:
            df["position"] = risk_budget.scale_positions(df["position"].to_numpy())

    X = df[feature_cols]
    groups = df["Symbol"] if "Symbol" in df.columns else df.get("SymbolCode")

    al_queue = ActiveLearningQueue()
    new_labels = al_queue.pop_labeled()
    if not new_labels.empty and label_cols:
        df = merge_labels(df, new_labels, label_cols[0])
        y = df[label_cols]
        X = df[feature_cols]
        save_history_parquet(df, root / "data" / "history.parquet")
    queue_stats = al_queue.stats()

    al_threshold = float(cfg.get("active_learning", {}).get("threshold", 0.6))

    if use_pseudo_labels and label_cols:
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
                if label_cols[0] not in df_pseudo.columns:
                    continue
                if not set(feature_cols).issubset(df_pseudo.columns):
                    continue
                X = pd.concat([X, df_pseudo[feature_cols]], ignore_index=True)
                y = pd.concat([y, df_pseudo[label_cols]], ignore_index=True)

    if "timestamp" in df.columns and len(df) == len(X):
        timestamps = _index_to_timestamps(df["timestamp"])
    else:
        timestamps = np.arange(len(X), dtype="int64")

    return DatasetBuildResult(
        df=df,
        X=X,
        y=y,
        features=feature_cols,
        label_cols=label_cols,
        abs_label_cols=abs_label_cols,
        vol_label_cols=vol_label_cols,
        groups=groups,
        timestamps=timestamps,
        al_queue=al_queue,
        al_threshold=al_threshold,
        queue_stats=queue_stats,
        risk_budget=risk_budget,
        user_budget=features.user_budget,
    )


def train_models(
    dataset: DatasetBuildResult,
    cfg: AppConfig,
    seed: int,
    model_type: str,
    use_multi_task_heads: bool,
    drift_monitor: ConceptDriftMonitor,
    donor_booster,
    use_focal: bool,
    fobj,
    feval,
    scaler_path: Path,
    root: Path,
    risk_target: dict | None,
    recorder: RunHistoryRecorder,
    resume_online: bool = False,
) -> TrainingResult:
    """Train the configured models and collect evaluation artefacts."""

    df = dataset.df
    X = dataset.X
    y = dataset.y
    features = list(dataset.features)
    label_cols = list(dataset.label_cols)
    abs_label_cols = list(dataset.abs_label_cols)
    vol_label_cols = list(dataset.vol_label_cols)
    groups = dataset.groups
    timestamps = dataset.timestamps
    al_queue = dataset.al_queue
    al_threshold = dataset.al_threshold
    risk_budget = dataset.risk_budget
    user_budget = dataset.user_budget

    queue_stats = dataset.queue_stats
    logger.info(
        "Active learning queue stats: pending=%s ready_for_merge=%s",
        queue_stats["awaiting_label"],
        queue_stats["ready_for_merge"],
    )
    try:  # pragma: no cover - mlflow optional
        mlflow.log_metric("al_queue_pending", queue_stats["awaiting_label"])
        mlflow.log_metric("al_queue_ready_for_merge", queue_stats["ready_for_merge"])
    except Exception:  # pragma: no cover - optional logging
        pass

    t_max = int(timestamps.max()) if len(timestamps) else 0

    if cfg.get("tune"):
        from tuning.tuning import tune_lightgbm

        def train_trial(params: dict, _trial) -> float:
            clf_params = {
                "n_estimators": 200,
                "n_jobs": cfg.get("n_jobs") or monitor.capabilities.cpus,
                "random_state": seed,
                **params,
            }
            if use_focal:
                clf_params["objective"] = "None"
            clf = LGBMClassifier(**clf_params)
            tscv_inner = PurgedTimeSeriesSplit(
                n_splits=cfg.get("n_splits", 5),
                embargo=cfg.get("max_horizon", 0),
            )
            scores: list[float] = []
            half_life = cfg.get("time_decay_half_life")
            balance = cfg.get("balance_classes")
            for tr_idx, va_idx in tscv_inner.split(X, groups=groups):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                fit_kwargs: dict[str, object] = {
                    "eval_set": [(X_va, y_va)],
                    "early_stopping_rounds": cfg.get("early_stopping_rounds", 50),
                    "verbose": False,
                }
                dq_w = dq_score_samples(X_tr)
                y_tr_arr = (
                    y_tr[label_cols[0]].to_numpy()
                    if label_cols
                    else y_tr.iloc[:, 0].to_numpy()
                )
                sw = combined_sample_weight(
                    y_tr_arr,
                    timestamps[tr_idx],
                    t_max,
                    balance,
                    half_life,
                    dq_w,
                )
                if sw is not None:
                    fit_kwargs["sample_weight"] = sw
                if use_focal:
                    fit_kwargs["fobj"] = fobj
                    fit_kwargs["feval"] = feval
                clf.fit(X_tr, y_tr, **fit_kwargs)
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
        return TrainingResult(
            final_pipe=None,
            features=features,
            aggregate_report={},
            boot_metrics={},
            base_f1=0.0,
            regime_thresholds={},
            overall_params=None,
            overall_q={},
            overall_cov=0.0,
            all_probs=[],
            all_conf=[],
            all_true=[],
            al_queue=al_queue,
            al_threshold=al_threshold,
            df=df,
            X=X,
            X_train_final=None,
            risk_budget=risk_budget,
            model_metadata={},
            f1_ci=(0.0, 0.0),
            prec_ci=(0.0, 0.0),
            rec_ci=(0.0, 0.0),
            final_score=0.0,
            should_log_artifacts=False,
        )

    if cfg.get("meta_init"):
        from analysis.meta_learning import (
            _LinearModel,
            fine_tune_model,
            load_meta_weights,
            save_meta_weights,
        )
        from models.meta_learner import steps_to
        from torch.utils.data import TensorDataset

        X_all = torch.tensor(df[features].values, dtype=torch.float32)
        y_all = torch.tensor(df[label_cols[0]].values, dtype=torch.float32)
        dataset_tensor = TensorDataset(X_all, y_all)
        state = load_meta_weights("lgbm")
        new_state, history = fine_tune_model(
            state, dataset_tensor, lambda: _LinearModel(len(features)), steps=5
        )
        logger.info("Meta-init adaptation steps: %s", steps_to(history))
        save_meta_weights(new_state, "lgbm", regime=cfg.get("symbol", "asset"))
        return TrainingResult(
            final_pipe=None,
            features=features,
            aggregate_report={},
            boot_metrics={},
            base_f1=0.0,
            regime_thresholds={},
            overall_params=None,
            overall_q={},
            overall_cov=0.0,
            all_probs=[],
            all_conf=[],
            all_true=[],
            al_queue=al_queue,
            al_threshold=al_threshold,
            df=df,
            X=X,
            X_train_final=None,
            risk_budget=risk_budget,
            model_metadata={},
            f1_ci=(0.0, 0.0),
            prec_ci=(0.0, 0.0),
            rec_ci=(0.0, 0.0),
            final_score=0.0,
            should_log_artifacts=False,
        )

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
            df.loc[mask, label_cols[0]].values,
            dtype=torch.float32,
        )
        dataset_tensor = TensorDataset(X_reg, y_reg)
        state = load_meta_weights("lgbm")
        new_state, _ = fine_tune_model(
            state, dataset_tensor, lambda: _LinearModel(len(features)), steps=5
        )
        save_meta_weights(new_state, "lgbm", regime=f"regime_{regime}")
        return TrainingResult(
            final_pipe=None,
            features=features,
            aggregate_report={},
            boot_metrics={},
            base_f1=0.0,
            regime_thresholds={},
            overall_params=None,
            overall_q={},
            overall_cov=0.0,
            all_probs=[],
            all_conf=[],
            all_true=[],
            al_queue=al_queue,
            al_threshold=al_threshold,
            df=df,
            X=X,
            X_train_final=None,
            risk_budget=risk_budget,
            model_metadata={},
            f1_ci=(0.0, 0.0),
            prec_ci=(0.0, 0.0),
            rec_ci=(0.0, 0.0),
            final_score=0.0,
            should_log_artifacts=False,
        )

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
            steps: list[tuple[str, object]] = [("sanitizer", _make_sanitizer(cfg))]
            if cfg.get("use_scaler", True):
                steps.append(("scaler", FeatureScaler.load(scaler_path)))
            clf_params = {
                "n_estimators": 200,
                "n_jobs": cfg.get("n_jobs") or monitor.capabilities.cpus,
                "random_state": seed,
                "keep_training_booster": True,
                **_lgbm_params(cfg),
            }
            if use_focal:
                clf_params["objective"] = "None"
            steps.append(("clf", LGBMClassifier(**clf_params)))
            pipe = Pipeline(steps)
        _register_clf(pipe.named_steps["clf"])
        sanitizer = pipe.named_steps.get("sanitizer")
        if sanitizer is not None and not getattr(sanitizer, "_fitted", False):
            sanitizer.fit(X)
        n_batches = math.ceil(len(X) / batch_size)
        half_life = cfg.get("time_decay_half_life")
        for batch_idx in range(start_batch, n_batches):
            start = batch_idx * batch_size
            end = min(len(X), start + batch_size)
            Xb = X.iloc[start:end]
            yb = y.iloc[start:end]
            dq_w = dq_score_samples(Xb)
            if sanitizer is not None:
                Xb = sanitizer.transform(Xb)
            if "scaler" in pipe.named_steps:
                scaler = pipe.named_steps["scaler"]
                if getattr(scaler, "group_col", None):
                    Xb_sym = Xb.copy()
                    Xb_sym[scaler.group_col] = groups.iloc[start:end].values
                    if hasattr(scaler, "partial_fit"):
                        scaler.partial_fit(Xb_sym)
                    Xb = scaler.transform(Xb_sym).drop(columns=[scaler.group_col])
                else:
                    if hasattr(scaler, "partial_fit"):
                        scaler.partial_fit(Xb)
                    Xb = scaler.transform(Xb)
            clf = pipe.named_steps["clf"]
            init_model = (
                donor_booster
                if batch_idx == 0 and donor_booster is not None and not ckpt
                else (clf.booster_ if (batch_idx > 0 or ckpt) else None)
            )
            fit_kwargs: dict[str, object] = {"init_model": init_model}
            if use_focal:
                fit_kwargs["fobj"] = fobj
                fit_kwargs["feval"] = feval
            yb_arr = (
                yb[label_cols[0]].to_numpy()
                if label_cols
                else yb.iloc[:, 0].to_numpy()
            )
            sw = combined_sample_weight(
                yb_arr,
                timestamps[start:end],
                t_max,
                cfg.get("balance_classes"),
                half_life,
                dq_w,
            )
            if sw is not None:
                fit_kwargs["sample_weight"] = sw
            clf.fit(Xb, yb, **fit_kwargs)
            state = {"model": pipe}
            if "scaler" in pipe.named_steps:
                state["scaler_state"] = pipe.named_steps["scaler"].state_dict()
            save_checkpoint(state, batch_idx, cfg.get("checkpoint_dir"))
        joblib.dump(pipe, root / "model.joblib")
        if "scaler" in pipe.named_steps:
            pipe.named_steps["scaler"].save(scaler_path)
        return TrainingResult(
            final_pipe=pipe,
            features=features,
            aggregate_report={},
            boot_metrics={},
            base_f1=0.0,
            regime_thresholds={},
            overall_params=None,
            overall_q={},
            overall_cov=0.0,
            all_probs=[],
            all_conf=[],
            all_true=[],
            al_queue=al_queue,
            al_threshold=al_threshold,
            df=df,
            X=X,
            X_train_final=None,
            risk_budget=risk_budget,
            model_metadata={},
            f1_ci=(0.0, 0.0),
            prec_ci=(0.0, 0.0),
            rec_ci=(0.0, 0.0),
            final_score=0.0,
            should_log_artifacts=False,
        )

    tscv = PurgedTimeSeriesSplit(
        n_splits=cfg.get("n_splits", 5), embargo=cfg.get("max_horizon", 0)
    )
    all_preds: list[int] = []
    all_regimes: list[int] = []
    all_true: list[int] = []
    all_probs: list[float] = []
    all_conf: list[float] = []
    all_lower: list[float] = []
    all_upper: list[float] = []
    risk_violation = False
    violation_penalty = 0.0
    all_residuals: dict[int, list[float]] = {}
    final_pipe: Pipeline | None = None
    X_train_final: pd.DataFrame | None = None
    last_val_X: pd.DataFrame | None = None
    last_val_y: np.ndarray | None = None
    last_val_probs: np.ndarray | None = None
    last_val_regimes: np.ndarray | None = None
    start_fold = 0
    ckpt = load_latest_checkpoint(cfg.get("checkpoint_dir"))
    if ckpt:
        last_fold, state = ckpt
        start_fold = last_fold + 1
        all_preds = state.get("all_preds", [])
        all_true = state.get("all_true", [])
        all_probs = state.get("all_probs", [])
        all_conf = state.get("all_conf", [])
        all_regimes = state.get("all_regimes", [])
        all_lower = state.get("all_lower", [])
        all_upper = state.get("all_upper", [])
        all_residuals = state.get("all_residuals", {})
        final_pipe = state.get("model")
        scaler_state = state.get("scaler_state")
        if final_pipe and scaler_state and "scaler" in final_pipe.named_steps:
            final_pipe.named_steps["scaler"].load_state_dict(scaler_state)
        logger.info("Resuming from checkpoint at fold %s", last_fold)

    last_split: tuple[np.ndarray, np.ndarray] | None = None
    half_life = cfg.get("time_decay_half_life")
    balance = cfg.get("balance_classes")
    interval_alpha = float(cfg.get("interval_alpha", 0.1))
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X, groups=groups)):
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
                X_aug = data["X"][..., -1, :]
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
            elif cfg.get("use_diffusion_aug", False):
                from analysis.scenario_diffusion import ScenarioDiffusion

                logger.info("Generating diffusion crash scenarios on the fly")
                seq_len = cfg.get("sequence_length", 50)
                n_syn = cfg.get("diffusion_aug_samples", len(X_train))
                model = ScenarioDiffusion(seq_len=seq_len)
                ret_idx = features.index("return") if "return" in features else None
                X_syn = np.zeros((n_syn, len(features)))
                y_syn = np.zeros(n_syn, dtype=int)
                for i in range(n_syn):
                    path = model.sample_crash_recovery(seq_len)
                    val = float(np.clip(path.min(), -0.3, 0.3))
                    if ret_idx is not None:
                        X_syn[i, ret_idx] = val
                df_aug = pd.DataFrame(X_syn, columns=features)
                X_train = pd.concat([X_train, df_aug], ignore_index=True)
                y_train = pd.concat(
                    [
                        y_train,
                        pd.Series(
                            y_syn, index=range(len(y_train), len(y_train) + len(y_syn))
                        ),
                    ],
                    ignore_index=True,
                )

        dq_w = dq_score_samples(X_train)
        if use_multi_task_heads:
            steps = [("sanitizer", _make_sanitizer(cfg))]
            if cfg.get("use_scaler", True):
                steps.append(("scaler", FeatureScaler()))
            hidden_default = 32
            if isinstance(X_train, pd.DataFrame):
                hidden_default = max(16, len(X_train.columns) // 2 or 16)
            head_cfg = {
                "classification_targets": label_cols,
                "abs_targets": abs_label_cols,
                "volatility_targets": vol_label_cols,
                "hidden_dim": int(cfg.get("head_hidden_dim", hidden_default)),
                "learning_rate": float(cfg.get("head_learning_rate", 0.01)),
                "epochs": int(cfg.get("head_epochs", 200)),
                "classification_weight": float(cfg.get("head_classification_weight", 1.0)),
                "abs_weight": float(cfg.get("head_abs_weight", 1.0)),
                "volatility_weight": float(cfg.get("head_volatility_weight", 1.0)),
                "l2": float(cfg.get("head_l2", 0.0)),
                "random_state": seed,
            }
            head = MultiTaskHeadEstimator(**head_cfg)
            steps.append(("multi_task", head))
            pipe = Pipeline(steps)
            fit_params = {}
            sw = combined_sample_weight(
                y_train[label_cols[0]].to_numpy()
                if label_cols
                else y_train.iloc[:, 0].to_numpy(),
                timestamps[train_idx],
                t_max,
                cfg.get("balance_classes"),
                half_life,
                dq_w,
            )
            if sw is not None:
                fit_params["multi_task__sample_weight"] = sw
            pipe.fit(X_train, y_train, **fit_params)
            if len(pipe.steps) > 1:
                X_val_shared = pipe[:-1].transform(X_val)
            else:
                X_val_shared = X_val
            if label_cols:
                class_probs = pipe.named_steps["multi_task"].predict_classification_proba(
                    X_val_shared
                )
                class_probs = np.asarray(class_probs, dtype=float)
                if class_probs.ndim == 1:
                    class_probs = class_probs.reshape(-1, 1)
                probs = class_probs[:, 0]
                val_probs = np.column_stack([1 - probs, probs])
            else:
                probs = np.zeros(len(X_val), dtype=float)
                val_probs = np.column_stack([1 - probs, probs])
            pipe.classification_targets_ = list(label_cols)
            pipe.abs_return_targets_ = list(abs_label_cols)
            pipe.volatility_targets_ = list(vol_label_cols)
            pipe.regression_feature_columns_ = (
                list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
            )
            pipe.head_config_ = dict(pipe.named_steps["multi_task"].head_config_)
            pipe.primary_label_ = pipe.named_steps["multi_task"].primary_label_
            pipe.classification_thresholds_ = dict(
                getattr(pipe.named_steps["multi_task"], "thresholds_", {})
            )
            heads = {}
            if abs_label_cols:
                heads["abs_return"] = {"type": "multi_task", "columns": list(abs_label_cols)}
            if vol_label_cols:
                heads["volatility"] = {"type": "multi_task", "columns": list(vol_label_cols)}
            pipe.regression_heads_ = heads
            pipe.regression_target_columns_ = {k: v["columns"] for k, v in heads.items()}
        else:
            steps = [("sanitizer", _make_sanitizer(cfg))]
            if model_type != "cross_modal" and cfg.get("use_scaler", True):
                steps.append(("scaler", FeatureScaler()))
            if model_type == "cross_modal":
                if CrossModalClassifier is None:
                    raise RuntimeError("CrossModalClassifier requires PyTorch")
                cm_cfg = cfg.get("cross_modal_model", {})
                estimator = CrossModalClassifier(
                    d_model=int(cm_cfg.get("d_model", cfg.get("d_model", 64))),
                    nhead=int(cm_cfg.get("nhead", cfg.get("nhead", 4))),
                    num_layers=int(cm_cfg.get("num_layers", cfg.get("num_layers", 2))),
                    dropout=float(cm_cfg.get("dropout", cfg.get("dropout", 0.1))),
                    lr=float(cm_cfg.get("lr", cm_cfg.get("learning_rate", 1e-3))),
                    epochs=int(cm_cfg.get("epochs", cm_cfg.get("num_epochs", 5))),
                    batch_size=int(cm_cfg.get("batch_size", 128)),
                    weight_decay=float(cm_cfg.get("weight_decay", 0.0)),
                    time_encoding=bool(
                        cm_cfg.get("time_encoding", cfg.get("time_encoding", False))
                    ),
                    average_attn_weights=bool(cm_cfg.get("average_attn_weights", True)),
                )
                steps.append(("clf", estimator))
                pipe = Pipeline(steps)
                fit_params_cm: dict[str, object] = {}
                sw = combined_sample_weight(
                    y_train[label_cols[0]].to_numpy()
                    if label_cols
                    else y_train.iloc[:, 0].to_numpy(),
                    timestamps[train_idx],
                    t_max,
                    cfg.get("balance_classes"),
                    half_life,
                    dq_w,
                )
                if sw is not None:
                    fit_params_cm["clf__sample_weight"] = sw
                pipe.fit(X_train, y_train, **fit_params_cm)
                val_probs = pipe.predict_proba(X_val)
                probs = val_probs[:, 1]
            else:
                clf_params = {
                    "n_estimators": 200,
                    "n_jobs": cfg.get("n_jobs") or monitor.capabilities.cpus,
                    "random_state": seed,
                    **_lgbm_params(cfg),
                }
                if use_focal:
                    clf_params["objective"] = "None"
                steps.append(("clf", LGBMClassifier(**clf_params)))
                pipe = Pipeline(steps)
                if "clf" in pipe.named_steps:
                    _register_clf(pipe.named_steps["clf"])
                fit_params = {"clf__eval_set": [(X_val, y_val)]}
                esr = cfg.get("early_stopping_rounds", 50)
                if esr:
                    fit_params["clf__early_stopping_rounds"] = esr
                if donor_booster is not None:
                    fit_params["clf__init_model"] = donor_booster
                sw = combined_sample_weight(
                    y_train[label_cols[0]].to_numpy()
                    if label_cols
                    else y_train.iloc[:, 0].to_numpy(),
                    timestamps[train_idx],
                    t_max,
                    cfg.get("balance_classes"),
                    half_life,
                    dq_w,
                )
                if sw is not None:
                    fit_params["clf__sample_weight"] = sw
                if use_focal:
                    fit_params["clf__fobj"] = fobj
                    fit_params["clf__feval"] = feval
                pipe.fit(X_train, y_train, **fit_params)
                val_probs = pipe.predict_proba(X_val)
                probs = val_probs[:, 1]

        sanitizer = pipe.named_steps.get("sanitizer")
        if sanitizer is not None and hasattr(sanitizer, "state_dict"):
            pipe.sanitizer_state_ = sanitizer.state_dict()
        pipe.head_config_ = getattr(pipe, "head_config_", {})
        pipe.classification_thresholds_ = getattr(
            pipe, "classification_thresholds_", {}
        )
        pipe.regression_heads_ = getattr(pipe, "regression_heads_", {})
        pipe.regression_target_columns_ = getattr(
            pipe, "regression_target_columns_", {}
        )
        if not hasattr(pipe, "regression_feature_columns_"):
            pipe.regression_feature_columns_ = (
                list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
            )
        if not hasattr(pipe, "classification_targets_"):
            pipe.classification_targets_ = list(label_cols)
        if not hasattr(pipe, "primary_label_"):
            pipe.primary_label_ = label_cols[0] if label_cols else None
        al_queue.push_low_confidence(X_val.index, val_probs, threshold=al_threshold)
        logger.info("Active learning queue size: %d", len(al_queue))
        for feat_row, pred in zip(X_val[features].to_dict("records"), probs):
            drift_monitor.update(feat_row, float(pred))
        regimes_val = X_val["market_regime"].values
        thr_dict, preds = find_regime_thresholds(y_val.values, probs, regimes_val)
        thr_dict = _normalise_regime_thresholds(thr_dict)
        for reg, thr_val in thr_dict.items():
            mlflow.log_metric(f"fold_{fold}_thr_regime_{reg}", float(thr_val))
        last_val_X, last_val_y, last_val_probs, last_val_regimes = (
            X_val,
            y_val,
            probs,
            regimes_val,
        )
        interval_params_fold, residuals, (lower, upper) = calibrate_intervals(
            y_val.values,
            probs,
            alpha=interval_alpha,
            regimes=regimes_val,
        )
        for reg in np.unique(regimes_val):
            mask = regimes_val == reg
            all_residuals.setdefault(int(reg), []).extend(
                [float(r) for r in residuals[mask]]
            )
        cov = float(interval_params_fold.coverage or 0.0)
        mlflow.log_metric(f"fold_{fold}_interval_coverage", cov)
        logger.info("Fold %d interval coverage: %.3f", fold, cov)
        conf = np.abs(probs - 0.5) * 2
        rp = cfg.strategy.risk_profile
        sizer = PositionSizer(
            capital=cfg.get("eval_capital", 1000.0) * rp.leverage_cap,
            target_vol=rp.tolerance,
        )
        for p, c in zip(probs, conf):
            sizer.size(p, confidence=c)
        all_conf.extend(conf)
        report = classification_report(y_val, preds, output_dict=True)
        logger.info("Fold %d\n%s", fold, classification_report(y_val, preds))
        budget_eval = risk_budget or user_budget
        if "return" in df.columns and budget_eval is not None:
            returns_val = df.loc[X_val.index, "return"].to_numpy()
            alpha = float(risk_target.get("cvar_level", 0.05)) if risk_target else 0.05
            cvar_val = float(-cvar(returns_val, alpha))
            mdd_val = float(max_drawdown(returns_val))
            mlflow.log_metric(f"fold_{fold}_cvar", cvar_val)
            mlflow.log_metric(f"fold_{fold}_max_drawdown", mdd_val)
            pen = risk_penalty(returns_val, budget_eval, level=alpha)
            if pen > 0:
                violation_penalty = float(pen)
                logger.warning(
                    "Risk constraints violated on fold %d: %.4f",
                    fold,
                    violation_penalty,
                )
                risk_violation = True
                break
        mlflow.log_metric(
            f"fold_{fold}_f1_weighted", report["weighted avg"]["f1-score"]
        )
        mlflow.log_metric(
            f"fold_{fold}_precision_weighted", report["weighted avg"]["precision"]
        )
        mlflow.log_metric(
            f"fold_{fold}_recall_weighted", report["weighted avg"]["recall"]
        )

        all_preds.extend(preds)
        all_true.extend(y_val)
        all_probs.extend(probs)
        all_regimes.extend(regimes_val)
        all_lower.extend(lower)
        all_upper.extend(upper)
        state = {
            "model": pipe,
            "all_preds": all_preds,
            "all_true": all_true,
            "all_probs": all_probs,
            "all_conf": all_conf,
            "all_regimes": all_regimes,
            "all_lower": all_lower,
            "all_upper": all_upper,
            "all_residuals": all_residuals,
            "metrics": report,
            "regime_thresholds": thr_dict,
        }
        state["interval_params"] = interval_params_fold.to_dict()
        if "scaler" in pipe.named_steps:
            state["scaler_state"] = pipe.named_steps["scaler"].state_dict()
        save_checkpoint(state, fold, cfg.get("checkpoint_dir"))

        if risk_violation:
            break

        if fold == tscv.n_splits - 1:
            final_pipe = pipe
            X_train_final = X_train
        last_split = (train_idx, val_idx)
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


def log_artifacts(
    training: TrainingResult,
    cfg: AppConfig,
    root: Path,
    export: bool,
    start_time: datetime,
    recorder: RunHistoryRecorder,
    df_unlabeled: pd.DataFrame | None = None,
) -> ArtifactLogResult:
    """Log ancillary artefacts after model training."""

    final_pipe = training.final_pipe
    features = training.features
    al_queue = training.al_queue
    al_threshold = training.al_threshold
    pseudo_path: Path | None = None
    queued_count = 0

    if final_pipe is not None and df_unlabeled is not None:
        X_unlabeled = df_unlabeled[features]
        y_unlabeled = (
            df_unlabeled["label"].values if "label" in df_unlabeled.columns else None
        )
        probs_unlabeled = final_pipe.predict_proba(X_unlabeled)
        queued_count = al_queue.push_low_confidence(
            X_unlabeled.index, probs_unlabeled, threshold=al_threshold
        )
        if queued_count:
            logger.info(
                "Queued %s low-confidence samples for review (threshold=%.2f)",
                queued_count,
                al_threshold,
            )
        queue_stats = al_queue.stats()
        logger.info(
            "Active learning queue stats: pending=%s ready_for_merge=%s",
            queue_stats["awaiting_label"],
            queue_stats["ready_for_merge"],
        )
        try:  # pragma: no cover - mlflow optional
            mlflow.log_metric("al_queue_pending", queue_stats["awaiting_label"])
            mlflow.log_metric(
                "al_queue_ready_for_merge", queue_stats["ready_for_merge"]
            )
        except Exception:  # pragma: no cover - optional logging
            pass
        pseudo_threshold = cfg.get("pseudo_label_threshold", 0.9)
        preds_unlabeled = probs_unlabeled.argmax(axis=1)
        max_probs = probs_unlabeled.max(axis=1)
        mask_high = max_probs >= pseudo_threshold
        pseudo_path = generate_pseudo_labels(
            final_pipe,
            X_unlabeled,
            y_true=y_unlabeled,
            threshold=pseudo_threshold,
            output_dir=root / "data" / "pseudo_labels",
            report_dir=root / "reports" / "pseudo_label",
        )
        high_conf_count = int(np.sum(mask_high))
        logger.info(
            "Generated pseudo labels at %s for %s high-confidence samples (thr=%.2f)",
            pseudo_path,
            high_conf_count,
            pseudo_threshold,
        )
        if high_conf_count:
            precision_est = float(max_probs[mask_high].mean())
            logger.info(
                "Pseudo label precision estimate=%.3f on %s samples",
                precision_est,
                high_conf_count,
            )
            try:  # pragma: no cover - mlflow optional
                mlflow.log_metric("pseudo_label_precision_estimate", precision_est)
            except Exception:
                pass
        if y_unlabeled is not None and high_conf_count:
            prec = precision_score(
                y_unlabeled[mask_high], preds_unlabeled[mask_high], zero_division=0
            )
            rec = recall_score(
                y_unlabeled[mask_high], preds_unlabeled[mask_high], zero_division=0
            )
            logger.info("Pseudo label precision=%s recall=%s", prec, rec)
            mlflow.log_metric("pseudo_label_precision", prec)
            mlflow.log_metric("pseudo_label_recall", rec)

    meta_features = pd.DataFrame({"prob": training.all_probs, "confidence": training.all_conf})
    meta_clf = train_meta_classifier(meta_features, training.all_true)
    meta_version_id = model_store.save_model(meta_clf, cfg, {"type": "meta"})
    training.model_metadata["meta_model_id"] = meta_version_id
    if final_pipe is not None:
        setattr(final_pipe, "model_metadata", training.model_metadata)
    perf = {
        "f1_weighted": training.boot_metrics["f1"],
        "precision_weighted": training.boot_metrics["precision"],
        "recall_weighted": training.boot_metrics["recall"],
        "f1_ci": [training.f1_ci[0], training.f1_ci[1]],
        "precision_ci": [training.prec_ci[0], training.prec_ci[1]],
        "recall_ci": [training.rec_ci[0], training.rec_ci[1]],
        "regime_thresholds": training.regime_thresholds,
        "meta_model_id": meta_version_id,
    }
    if training.overall_params is not None:
        perf["interval"] = training.overall_params.to_dict()
        perf["interval_q"] = training.overall_q
        perf["interval_alpha"] = cfg.get("interval_alpha", 0.1)
        perf["interval_coverage"] = training.overall_cov
        if training.overall_params.coverage_by_regime:
            perf["interval_coverage_by_regime"] = {
                int(k): float(v)
                for k, v in training.overall_params.coverage_by_regime.items()
            }
        if final_pipe is not None:
            setattr(final_pipe, "interval_q", training.overall_params.quantiles)
            setattr(final_pipe, "interval_params", training.overall_params)
            setattr(final_pipe, "interval_coverage", training.overall_params.coverage)
            setattr(final_pipe, "interval_alpha", training.overall_params.alpha)
            if training.overall_params.coverage_by_regime:
                setattr(
                    final_pipe,
                    "interval_coverage_by_regime",
                    {
                        int(k): float(v)
                        for k, v in training.overall_params.coverage_by_regime.items()
                    },
                )
    version_id = persist_model(
        final_pipe,
        cfg,
        perf,
        features=features,
        root=root,
    )
    try:  # pragma: no cover - mlflow optional
        mlflow.log_artifact(str(root / "model.joblib"))
    except Exception:
        pass
    logger.info("Registered model version %s", version_id)

    regimes_path = root / "models" / "regime_models"
    regimes_path.mkdir(parents=True, exist_ok=True)
    recorder.add_artifact(
        regimes_path, dest_name="models/regime_models", optional=True
    )
    logger.info("Regime-specific models saved to %s", regimes_path)

    out = root / "classification_report.json"
    with out.open("w") as f:
        json.dump(training.aggregate_report, f, indent=2)
    mlflow.log_artifact(str(out))
    recorder.add_artifact(
        out, dest_name="reports/classification_report.json", optional=True
    )

    if (
        final_pipe is not None
        and training.X_train_final is not None
        and cfg.get("feature_importance", False)
    ):
        report_dir = Path(__file__).resolve().parent / "reports"
        paths = generate_shap_report(
            final_pipe,
            training.X_train_final[features],
            report_dir,
        )
        for p in paths.values():
            mlflow.log_artifact(str(p))

    if export and final_pipe is not None:
        from models.export import export_lightgbm

        sample = training.X.iloc[: min(len(training.X), 10)]
        if "scaler" in final_pipe.named_steps:
            sample = final_pipe.named_steps["scaler"].transform(sample)
        clf = final_pipe.named_steps.get("clf", final_pipe)
        export_lightgbm(clf, sample)

    logger.info("Active learning queue size: %d", len(al_queue))
    card_md, card_json = model_card.generate(
        cfg,
        [root / "data" / "history.parquet"],
        features,
        training.aggregate_report,
        root / "reports" / "model_cards",
    )
    recorder.add_artifact(card_md, dest_name="reports/model_card.md", optional=True)
    recorder.add_artifact(
        card_json, dest_name="reports/model_card.json", optional=True
    )
    mlflow.log_metric(
        "runtime",
        (datetime.now() - start_time).total_seconds(),
    )

    return ArtifactLogResult(
        meta_model_id=meta_version_id,
        model_version_id=version_id,
        pseudo_label_path=pseudo_path,
        queued_count=queued_count,
    )


def run_training_pipeline(
    cfg: AppConfig | dict | None = None,
    export: bool = False,
    resume_online: bool = False,
    df_override: pd.DataFrame | None = None,
    transfer_from: str | None = None,
    use_pseudo_labels: bool = False,
    df_unlabeled: pd.DataFrame | None = None,
    risk_target: dict | None = None,
) -> float:
    ensure_environment()
    if cfg is None:
        cfg = load_config()
    elif isinstance(cfg, dict):
        cfg = AppConfig(**cfg)
    model_type = str(cfg.training.model_type or "lgbm").lower()
    cfg.training.model_type = model_type
    if cfg.training.use_pseudo_labels:
        use_pseudo_labels = True

    config_payload = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg
    recorder_extra = {
        "resume_online": bool(resume_online),
        "export": bool(export),
        "transfer_from": transfer_from,
        "risk_target": risk_target,
    }
    recorder = RunHistoryRecorder(
        component="training.pipeline",
        config=config_payload,
        tags={
            "model_type": model_type,
            "use_pseudo_labels": bool(use_pseudo_labels),
        },
        extra={k: v for k, v in recorder_extra.items() if v is not None},
    )
    recorder.add_artifact(LOG_DIR / "app.log", dest_name="logs/app.log", optional=True)
    recorder.add_artifact(
        LOG_DIR / "trades.csv", dest_name="logs/trades.csv", optional=True
    )
    recorder.start()
    previous_run_id = os.environ.get("MT5_RUN_ID")
    os.environ["MT5_RUN_ID"] = recorder.run_id
    status = "failed"
    result_value: float | None = None
    error: Exception | None = None
    base_f1 = 0.0

    _subscribe_cpu_updates(cfg)
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    recorder.update_context(seed=seed)
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)
    scaler_path = root / "scaler.pkl"
    start_time = datetime.now()

    drift_monitor = ConceptDriftMonitor(
        method=cfg.training.drift_method,
        delta=float(cfg.training.drift_delta),
    )

    donor_booster = _load_donor_booster(transfer_from) if transfer_from else None

    use_focal = cfg.training.use_focal_loss
    focal_alpha = cfg.training.focal_alpha
    focal_gamma = cfg.training.focal_gamma
    fobj = make_focal_loss(focal_alpha, focal_gamma) if use_focal else None
    feval = make_focal_loss_metric(focal_alpha, focal_gamma) if use_focal else None

    mlflow_run_active = False
    try:
        mlflow.start_run("training", cfg)
        mlflow_run_active = True
    except Exception:  # pragma: no cover - mlflow optional
        mlflow_run_active = False
    try:
        try:  # pragma: no cover - mlflow optional
            mlflow.log_param("model_type", model_type)
        except Exception:
            pass

        history = load_histories(cfg, root, df_override=df_override)
        recorder.update_context(
            validate=history.validate,
            stream_enabled=history.stream,
            stream_chunk_size=history.chunk_size,
            stream_feature_lookback=history.feature_lookback,
            data_source=history.data_source,
        )
        try:  # pragma: no cover - mlflow optional
            mlflow.log_param("data_source", history.data_source)
        except Exception:
            pass
        if history.stream_metadata:
            metadata = history.stream_metadata
            try:  # pragma: no cover - mlflow optional
                if metadata.get("chunk_size") is not None:
                    mlflow.log_param("stream_chunk_size", int(metadata["chunk_size"]))
                if metadata.get("feature_lookback") is not None:
                    mlflow.log_param(
                        "stream_feature_lookback",
                        int(metadata["feature_lookback"]),
                    )
            except Exception:
                pass
            recorder.update_context(stream_metadata=metadata)

        features_state = prepare_features(history, cfg, model_type, root)
        dataset = build_datasets(
            features_state,
            cfg,
            root,
            use_pseudo_labels=use_pseudo_labels,
            risk_target=risk_target,
        )
        training_result = train_models(
            dataset,
            cfg,
            seed,
            model_type,
            features_state.use_multi_task_heads,
            drift_monitor,
            donor_booster,
            use_focal,
            fobj,
            feval,
            scaler_path,
            root,
            risk_target,
            recorder,
            resume_online=resume_online,
        )
        base_f1 = training_result.final_score
        result_value = training_result.final_score
        status = "completed"
        if training_result.should_log_artifacts:
            log_artifacts(
                training_result,
                cfg,
                root,
                export,
                start_time,
                recorder,
                df_unlabeled=df_unlabeled,
            )
        return float(training_result.final_score)
    except Exception as exc:
        error = exc
        recorder.add_error(str(exc))
        raise
    finally:
        if mlflow_run_active:
            try:
                mlflow.end_run()
            except Exception:
                pass
        _prune_finished_classifiers()
        if previous_run_id is not None:
            os.environ["MT5_RUN_ID"] = previous_run_id
        else:
            os.environ.pop("MT5_RUN_ID", None)
        if error is None and status == "failed":
            status = "completed"
        recorder.finish(status=status, error=error, result=result_value)
    return float(base_f1)


def _run_training(
    cfg: AppConfig | dict | None = None,
    export: bool = False,
    resume_online: bool = False,
    df_override: pd.DataFrame | None = None,
    transfer_from: str | None = None,
    use_pseudo_labels: bool = False,
    df_unlabeled: pd.DataFrame | None = None,
    risk_target: dict | None = None,
) -> float:
    """Backward-compatible wrapper for the training pipeline."""

    return run_training_pipeline(
        cfg=cfg,
        export=export,
        resume_online=resume_online,
        df_override=df_override,
        transfer_from=transfer_from,
        use_pseudo_labels=use_pseudo_labels,
        df_unlabeled=df_unlabeled,
        risk_target=risk_target,
    )


@log_exceptions
def main(
    cfg: AppConfig | dict | None = None,
    export: bool = False,
    resume_online: bool = False,
    df_override: pd.DataFrame | None = None,
    transfer_from: str | None = None,
    use_pseudo_labels: bool = False,
    df_unlabeled: pd.DataFrame | None = None,
    risk_target: dict | None = None,
) -> float:
    """Train LightGBM model and return weighted F1 score.

    Parameters
    ----------
    risk_target:
        Optional dictionary specifying risk constraints. Supported keys are
        ``"max_drawdown"`` and ``"cvar"`` (expected shortfall). When provided,
        the final score is penalised if the constraints are violated.
    """

    init_logging()
    monitor.start()
    start_capability_watch()
    try:
        return run_training_pipeline(
            cfg=cfg,
            export=export,
            resume_online=resume_online,
            df_override=df_override,
            transfer_from=transfer_from,
            use_pseudo_labels=use_pseudo_labels,
            df_unlabeled=df_unlabeled,
            risk_target=risk_target,
        )
    finally:
        monitor.stop()


def launch(
    cfg: AppConfig | dict | None = None,
    export: bool = False,
    resume_online: bool = False,
    transfer_from: str | None = None,
    use_pseudo_labels: bool = False,
    risk_target: dict | None = None,
) -> list[float]:
    """Launch training locally or across a Ray cluster."""
    if cfg is None:
        cfg = load_config()
    elif isinstance(cfg, dict):
        cfg = AppConfig(**cfg)
    curriculum_cfg = cfg.get("curriculum") if hasattr(cfg, "get") else None
    if curriculum_cfg:

        def _build_fn(stage_cfg: dict) -> callable:
            def _run_stage() -> float:
                base = cfg.model_dump()
                base.update(stage_cfg.get("config", {}))
                stage_cfg_obj = AppConfig(**base)
                return main(
                    stage_cfg_obj,
                    export=export,
                    resume_online=resume_online,
                    transfer_from=transfer_from,
                    use_pseudo_labels=use_pseudo_labels,
                    risk_target=risk_target,
                )

            return _run_stage

        scheduler = CurriculumScheduler.from_config(curriculum_cfg, _build_fn)
        if scheduler is not None:
            scheduler.run()
            # return metrics from each stage for transparency
            return [m for _, m in scheduler.metrics]
    if cluster_available():
        seeds = cfg.get("seeds", [cfg.training.seed])
        results = []
        for s in seeds:
            cfg_s = cfg.model_dump()
            cfg_s["training"]["seed"] = s
            results.append(
                submit(
                    main,
                    cfg_s,
                    export=export,
                    resume_online=resume_online,
                    transfer_from=transfer_from,
                    use_pseudo_labels=use_pseudo_labels,
                    risk_target=risk_target,
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
            risk_target=risk_target,
        )
    ]


