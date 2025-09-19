import time
from pathlib import Path
import random
import logging
from typing import Any, Iterable, Tuple
import numpy as np
import pandas as pd
import joblib
import json
from river import compose, preprocessing, linear_model
from sklearn.metrics import f1_score

from utils import load_config
from log_utils import setup_logging, log_exceptions
from data.live_recorder import load_ticks
from data.features import make_features
from data.labels import triple_barrier
from train_utils import resolve_training_features
from model_registry import save_model
from state_manager import watch_config
from analysis.regime_thresholds import find_regime_thresholds

logger = logging.getLogger(__name__)


def init_logging() -> logging.Logger:
    """Initialise structured logging for the online training service."""

    setup_logging()
    return logging.getLogger(__name__)


@log_exceptions
def train_online(
    data_path: Path | str | None = None,
    model_dir: Path | str | None = None,
    *,
    min_ticks: int = 1000,
    interval: int = 300,
    run_once: bool = False,
) -> None:
    """Incrementally update a river model using recorded live ticks.

    Parameters
    ----------
    data_path: Path, optional
        Location of the recorded tick dataset. Defaults to ``data/live``.
    model_dir: Path, optional
        Directory where models are stored. Defaults to ``models``.
    min_ticks: int, optional
        Minimum number of new ticks required to trigger a training step.
    interval: int, optional
        Maximum number of seconds between training steps.
    run_once: bool, optional
        When ``True`` the function processes a single training step and
        returns immediately. Useful for tests.
    """

    init_logging()
    cfg = load_config()
    _observer = watch_config(cfg)
    try:
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        training_cfg = getattr(cfg, "training", None)

        def _cfg_value(name: str, default: Any) -> Any:
            if training_cfg is not None and hasattr(training_cfg, name):
                value = getattr(training_cfg, name)
            else:
                value = cfg.get(name, None)
            return default if value is None else value

        label_col = "tb_label"
        pt_mult = float(_cfg_value("pt_mult", 0.01))
        sl_mult = float(_cfg_value("sl_mult", 0.01))
        horizon_raw = int(_cfg_value("max_horizon", 5))
        label_horizon = max(1, horizon_raw)

        window_default = max(512, label_horizon * 2)
        try:
            feature_window = int(cfg.get("online_feature_window", window_default))
        except (TypeError, ValueError):
            feature_window = window_default
        feature_window = max(feature_window, label_horizon * 2)

        validation_default = max(256, label_horizon * 4)
        try:
            validation_window = int(cfg.get("online_validation_window", validation_default))
        except (TypeError, ValueError):
            validation_window = validation_default
        validation_window = max(1, validation_window)
        min_training_samples = max(32, label_horizon * 2)

        def _normalise_regime_thresholds(
            thresholds: dict[Any, Any] | None,
        ) -> dict[int, float]:
            if not thresholds:
                return {}
            normalised: dict[int, float] = {}
            for key, value in thresholds.items():
                try:
                    norm_key = int(key)
                except (TypeError, ValueError):
                    continue
                try:
                    normalised[norm_key] = float(value)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
            return normalised

        def _best_f1_threshold(
            y_true: Iterable[int], probabilities: Iterable[float]
        ) -> Tuple[float, float]:
            probs = np.asarray(list(probabilities), dtype=float)
            if probs.size == 0:
                return 0.5, 0.0
            y = np.asarray(list(y_true), dtype=int)
            candidate_thresholds = np.unique(
                np.concatenate(
                    (
                        np.linspace(0.01, 0.99, 99),
                        np.clip(probs, 0.0, 1.0),
                        np.array([0.5]),
                    )
                )
            )
            best_threshold = 0.5
            best_f1 = -1.0
            for threshold in candidate_thresholds:
                preds = (probs >= threshold).astype(int)
                score = f1_score(y, preds, zero_division=0)
                if score > best_f1 or (
                    np.isclose(score, best_f1) and threshold < best_threshold
                ):
                    best_f1 = float(score)
                    best_threshold = float(threshold)
            return best_threshold, best_f1

        root = Path(__file__).resolve().parent
        data_path = Path(data_path) if data_path is not None else root / "data" / "live"
        model_dir = Path(model_dir) if model_dir is not None else root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        latest_path = model_dir / "online_latest.joblib"

        model = compose.Pipeline(
            preprocessing.StandardScaler(), linear_model.LogisticRegression()
        )

        def _ensure_timestamp(value: Any) -> pd.Timestamp | None:
            if value is None or isinstance(value, pd.Timestamp):
                return value
            try:
                return pd.Timestamp(value)
            except Exception:
                return None

        def _sanitize_timings(timings: Any) -> dict[str, Any]:
            cleaned: dict[str, Any] = {}
            if not isinstance(timings, dict):
                return cleaned
            for name, entry in timings.items():
                if isinstance(entry, dict):
                    cleaned[name] = {
                        key: float(val)
                        if isinstance(val, (int, float, np.floating))
                        else val
                        for key, val in entry.items()
                    }
                elif isinstance(entry, (int, float, np.floating)):
                    cleaned[name] = float(entry)
                else:
                    cleaned[name] = entry
            return cleaned

        def _build_feature_metadata(
            columns: list[str] | None,
            timings: Any,
            last_tick: pd.Timestamp | None,
            last_train: pd.Timestamp | None,
            *,
            filled: list[str] | None = None,
        ) -> dict[str, Any]:
            meta: dict[str, Any] = {
                "feature_columns": list(columns or []),
                "label_column": label_col,
                "feature_window": int(feature_window),
            }
            sanitized = _sanitize_timings(timings)
            if sanitized:
                meta["feature_timings"] = sanitized
            if last_tick is not None:
                meta["last_tick_ts"] = pd.Timestamp(last_tick).isoformat()
            if last_train is not None:
                meta["last_train_ts"] = pd.Timestamp(last_train).isoformat()
            if filled:
                meta["filled_missing_columns"] = list(filled)
            return meta

        last_tick_ts: pd.Timestamp | None = None
        last_train_ts: pd.Timestamp | None = None
        feature_columns: list[str] | None = None
        feature_metadata: dict[str, Any] = {}
        history = pd.DataFrame()
        best_threshold: float | None = None
        validation_f1: float | None = None
        regime_thresholds: dict[int, float] = {}

        if latest_path.exists():
            try:
                payload = joblib.load(latest_path)
                if isinstance(payload, dict):
                    model = payload.get("model", model)
                    last_tick_ts = _ensure_timestamp(payload.get("last_tick_ts"))
                    last_train_ts = _ensure_timestamp(payload.get("last_train_ts"))
                    stored_cols = payload.get("feature_columns")
                    if stored_cols:
                        feature_columns = list(stored_cols)
                    feature_metadata = payload.get("feature_metadata", {}) or {}
                    stored_threshold = payload.get("best_threshold")
                    if stored_threshold is not None:
                        try:
                            best_threshold = float(stored_threshold)
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            best_threshold = None
                    stored_regime = _normalise_regime_thresholds(
                        payload.get("regime_thresholds")
                    )
                    if stored_regime:
                        regime_thresholds = stored_regime
                    stored_f1 = payload.get("validation_f1")
                    if stored_f1 is not None:
                        try:
                            validation_f1 = float(stored_f1)
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            validation_f1 = None
                else:
                    model, last_tick_ts = payload
                    last_train_ts = last_tick_ts
                logger.info("Loaded existing model from %s", latest_path)
            except Exception as exc:  # pragma: no cover - just warn
                logger.warning("Failed to load existing model: %s", exc)
                last_tick_ts = None
                last_train_ts = None
                feature_columns = None
                feature_metadata = {}
        if best_threshold is not None:
            setattr(model, "best_threshold", best_threshold)
            setattr(model, "best_threshold_", best_threshold)
        if validation_f1 is not None:
            setattr(model, "validation_f1_", validation_f1)
        if regime_thresholds:
            setattr(model, "regime_thresholds", dict(regime_thresholds))
            setattr(model, "regime_thresholds_", dict(regime_thresholds))

        last_train = time.time()
        new_ticks = 0
        start_ts = last_train_ts or last_tick_ts
        buffer_size = max(feature_window + label_horizon, label_horizon * 2)

        while True:
            df = load_ticks(data_path, last_tick_ts)
            if not df.empty:
                df = df.sort_values("Timestamp").reset_index(drop=True)
                history = pd.concat([history, df], ignore_index=True)
                history = history.sort_values("Timestamp").reset_index(drop=True)
                if len(history) > buffer_size:
                    history = history.iloc[-buffer_size:].reset_index(drop=True)
                if not history.empty and "Timestamp" in history.columns:
                    start_ts = history["Timestamp"].min()
                    last_tick_ts = history["Timestamp"].max()

                base_frame = history.copy()
                if {"Bid", "Ask"}.issubset(base_frame.columns) and "mid" not in base_frame.columns:
                    base_frame["mid"] = (base_frame["Bid"] + base_frame["Ask"]) / 2

                try:
                    feature_frame = make_features(base_frame.copy(), validate=False)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Feature generation failed: %s", exc, exc_info=True)
                    feature_frame = pd.DataFrame()

                if not feature_frame.empty:
                    if "Timestamp" in base_frame.columns:
                        feature_frame["Timestamp"] = base_frame["Timestamp"]
                    if "Symbol" in base_frame.columns and "Symbol" not in feature_frame.columns:
                        feature_frame["Symbol"] = base_frame["Symbol"]
                    if "mid" not in feature_frame.columns and "mid" in base_frame.columns:
                        feature_frame["mid"] = base_frame["mid"]

                    mid_series = feature_frame.get("mid")
                    if mid_series is None:
                        logger.debug("Skipping update because 'mid' price is missing from features")
                    else:
                        labels = triple_barrier(mid_series, pt_mult, sl_mult, label_horizon)
                        feature_frame[label_col] = labels
                        if label_horizon > 0 and len(feature_frame) > label_horizon:
                            train_frame = feature_frame.iloc[:-label_horizon].copy()
                        else:
                            train_frame = feature_frame.copy()

                        if last_train_ts is not None and "Timestamp" in train_frame.columns:
                            train_frame = train_frame[train_frame["Timestamp"] > last_train_ts]

                        train_frame = train_frame.replace([np.inf, -np.inf], np.nan)
                        train_frame.dropna(subset=[label_col], inplace=True)

                        if not train_frame.empty:
                            if feature_columns is None:
                                try:
                                    training_section = getattr(cfg, "training", cfg)
                                    feature_columns = resolve_training_features(
                                        train_frame,
                                        train_frame[label_col],
                                        training_section,
                                        id_columns={"Timestamp", "Symbol"},
                                        target_columns={label_col},
                                    )
                                except Exception as exc:  # pragma: no cover - best effort
                                    logger.warning(
                                        "Falling back to numeric feature set: %s",
                                        exc,
                                    )
                                    numeric_cols = train_frame.select_dtypes(
                                        include=[np.number, "bool"]
                                    ).columns
                                    skip_cols = {label_col, "Timestamp", "Symbol"}
                                    feature_columns = [
                                        col
                                        for col in train_frame.columns
                                        if col in numeric_cols and col not in skip_cols
                                    ]

                            updates = 0
                            filled_missing: list[str] = []
                            if feature_columns:
                                feature_columns = list(feature_columns)
                                setattr(model, "training_features_", list(feature_columns))
                                for col in feature_columns:
                                    if col not in train_frame.columns:
                                        train_frame[col] = 0.0
                                        filled_missing.append(col)

                                train_frame = train_frame.copy()
                                train_frame[feature_columns] = train_frame[
                                    feature_columns
                                ].fillna(0.0)

                                validation_rows = 0
                                if len(train_frame) > min_training_samples:
                                    validation_rows = min(
                                        validation_window,
                                        len(train_frame) - min_training_samples,
                                    )
                                if validation_rows > 0:
                                    val_frame = train_frame.tail(validation_rows).copy()
                                    train_subset = train_frame.iloc[:-validation_rows].copy()
                                else:
                                    val_frame = pd.DataFrame()
                                    train_subset = train_frame.copy()

                                def _prepare_feature_dict(row: pd.Series) -> dict[str, Any]:
                                    feats = row[feature_columns].to_dict()
                                    return {
                                        key: float(val)
                                        if isinstance(val, (np.generic, float, int, bool))
                                        else val
                                        for key, val in feats.items()
                                    }

                                for _, row in train_subset.iterrows():
                                    feature_values = _prepare_feature_dict(row)
                                    y_val = 1 if row[label_col] > 0 else 0
                                    model.learn_one(feature_values, y_val)
                                    timestamp = row.get("Timestamp")
                                    if timestamp is not None:
                                        last_train_ts = pd.Timestamp(timestamp)
                                    updates += 1

                                predict_one = getattr(model, "predict_proba_one", None)
                                validation_samples: list[
                                    tuple[dict[str, Any], int, Any, pd.Timestamp | None]
                                ] = []
                                if (
                                    callable(predict_one)
                                    and not val_frame.empty
                                    and label_col in val_frame.columns
                                ):
                                    val_probs: list[float] = []
                                    val_true: list[int] = []
                                    val_regimes: list[Any] = []
                                    val_ts: list[pd.Timestamp | None] = []
                                    for _, row in val_frame.iterrows():
                                        features_dict = _prepare_feature_dict(row)
                                        y_val = 1 if row[label_col] > 0 else 0
                                        try:
                                            prob = float(
                                                predict_one(features_dict).get(1, 0.0)
                                            )
                                        except Exception:  # pragma: no cover - best effort
                                            logger.exception(
                                                "Online validation prediction failed"
                                            )
                                            val_probs = []
                                            break
                                        val_probs.append(prob)
                                        val_true.append(y_val)
                                        val_regimes.append(row.get("market_regime"))
                                        ts_val = row.get("Timestamp")
                                        val_ts.append(
                                            pd.Timestamp(ts_val)
                                            if ts_val is not None
                                            else None
                                        )
                                        validation_samples.append(
                                            (features_dict, y_val, row.get("market_regime"), val_ts[-1])
                                        )
                                    if val_probs:
                                        probs_arr = np.asarray(val_probs, dtype=float)
                                        true_arr = np.asarray(val_true, dtype=int)
                                        best_thr, val_f1 = _best_f1_threshold(
                                            true_arr, probs_arr
                                        )
                                        preds_arr = (probs_arr >= best_thr).astype(int)
                                        regimes_series = pd.Series(val_regimes)
                                        regime_thresholds_map: dict[int, float] = {}
                                        if regimes_series.notna().any():
                                            valid_mask = regimes_series.notna().to_numpy()
                                            try:
                                                regime_ids = regimes_series[valid_mask].astype(int).to_numpy()
                                                regime_thresholds_map, regime_preds = (
                                                    find_regime_thresholds(
                                                        true_arr[valid_mask],
                                                        probs_arr[valid_mask],
                                                        regime_ids,
                                                    )
                                                )
                                                preds_arr[valid_mask] = regime_preds
                                            except Exception:  # pragma: no cover - defensive
                                                logger.exception(
                                                    "Failed to compute regime thresholds"
                                                )
                                                regime_thresholds_map = {}
                                        try:
                                            validation_f1 = float(
                                                f1_score(true_arr, preds_arr, zero_division=0)
                                            )
                                        except Exception:  # pragma: no cover - defensive
                                            validation_f1 = float(val_f1)
                                        best_threshold = float(best_thr)
                                        if regime_thresholds_map:
                                            regime_thresholds = {
                                                int(k): float(v)
                                                for k, v in regime_thresholds_map.items()
                                            }
                                        else:
                                            regime_thresholds = {0: float(best_thr)}
                                        setattr(model, "best_threshold", best_threshold)
                                        setattr(model, "best_threshold_", best_threshold)
                                        setattr(model, "validation_f1_", validation_f1)
                                        setattr(model, "regime_thresholds", dict(regime_thresholds))
                                        setattr(
                                            model,
                                            "regime_thresholds_",
                                            dict(regime_thresholds),
                                        )
                                    else:
                                        validation_samples = []

                                for feat_values, y_val, _, ts_val in validation_samples:
                                    model.learn_one(feat_values, y_val)
                                    if ts_val is not None:
                                        last_train_ts = pd.Timestamp(ts_val)
                                    updates += 1

                                if updates:
                                    new_ticks += updates
                                    feature_metadata = _build_feature_metadata(
                                        feature_columns,
                                        feature_frame.attrs.get("feature_timings"),
                                        last_tick_ts,
                                        last_train_ts,
                                        filled=filled_missing,
                                    )

            now = time.time()
            if new_ticks >= min_ticks or (now - last_train) >= interval:
                if new_ticks > 0:
                    state = {
                        "model": model,
                        "last_tick_ts": last_tick_ts,
                        "last_train_ts": last_train_ts,
                        "feature_columns": list(feature_columns or []),
                        "feature_metadata": feature_metadata,
                        "label_column": label_col,
                    }
                    if best_threshold is not None:
                        state["best_threshold"] = float(best_threshold)
                    if validation_f1 is not None:
                        state["validation_f1"] = float(validation_f1)
                    if regime_thresholds:
                        state["regime_thresholds"] = dict(regime_thresholds)
                    joblib.dump(state, latest_path)
                    version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                    preprocessing_meta = dict(feature_metadata)
                    if feature_columns is not None and "feature_columns" not in preprocessing_meta:
                        preprocessing_meta["feature_columns"] = list(feature_columns)
                    preprocessing_meta.setdefault("label_column", label_col)
                    preprocessing_meta.setdefault("feature_window", feature_window)
                    if last_tick_ts is not None:
                        preprocessing_meta.setdefault("last_tick_ts", last_tick_ts.isoformat())
                    if last_train_ts is not None:
                        preprocessing_meta.setdefault("last_train_ts", last_train_ts.isoformat())
                    performance_meta: dict[str, Any] = {}
                    if validation_f1 is not None:
                        performance_meta["validation_f1"] = float(validation_f1)
                    if best_threshold is not None:
                        performance_meta["best_threshold"] = float(best_threshold)
                    if regime_thresholds:
                        performance_meta["regime_thresholds"] = {
                            str(k): float(v) for k, v in regime_thresholds.items()
                        }
                    metadata = {
                        "drawdown_limit": cfg.get("drawdown_limit"),
                        "training_window": [
                            start_ts.isoformat() if start_ts else None,
                            last_train_ts.isoformat() if last_train_ts else None,
                        ],
                        "preprocessing": preprocessing_meta,
                    }
                    if performance_meta:
                        metadata["performance"] = performance_meta
                    save_path = model_dir / f"online_{version}.pkl"
                    save_model(
                        f"online_{version}",
                        model,
                        metadata,
                        save_path,
                    )
                    prov_log = model_dir / "training_provenance.log"
                    try:
                        with prov_log.open("a", encoding="utf-8") as f:
                            record = {"version": version, **metadata}
                            f.write(json.dumps(record) + "\n")
                    except Exception:  # pragma: no cover - best effort
                        logger.debug("Failed to write provenance to %s", prov_log)
                    feature_count = len(preprocessing_meta.get("feature_columns", []))
                    logger.info(
                        "Updated model %s with %d new ticks; window %s - %s drawdown_limit=%s feature_count=%d",
                        version,
                        new_ticks,
                        metadata["training_window"][0],
                        metadata["training_window"][1],
                        metadata["drawdown_limit"],
                        feature_count,
                    )
                    new_ticks = 0
                    last_train = now
                if run_once:
                    break
            if run_once:
                break
            time.sleep(1)
    finally:
        try:
            _observer.stop()
        except Exception:
            logger.exception("Failed to stop config watcher")
        try:
            _observer.join()
        except Exception:
            logger.exception("Failed to join config watcher")


def rollback_model(
    version: str | None = None, model_dir: Path | str | None = None
) -> Path:
    """Restore a previously saved model version as the active one.

    Parameters
    ----------
    version:
        Timestamp identifier of the model to restore (e.g. ``20240101000000``).
        When omitted the function rolls back to the second most recent model.
    model_dir:
        Directory where model artifacts are stored. Defaults to ``models``.
    """

    root = Path(__file__).resolve().parent
    model_dir = Path(model_dir) if model_dir is not None else root / "models"
    versions = sorted(model_dir.glob("online_*.pkl"))
    if version:
        candidate = model_dir / f"online_{version}.pkl"
        if not candidate.exists():  # pragma: no cover - user input
            raise FileNotFoundError(candidate)
    else:
        if len(versions) < 2:
            raise ValueError("No previous model available for rollback")
        candidate = versions[-2]
    model = joblib.load(candidate)
    meta_path = candidate.with_suffix(".json")
    last_tick_ts: pd.Timestamp | None = None
    last_train_ts: pd.Timestamp | None = None
    feature_columns: list[str] = []
    feature_metadata: dict[str, Any] = {}
    performance_meta: dict[str, Any] = {}
    label_col = "tb_label"
    if meta_path.exists():
        try:
            info = json.loads(meta_path.read_text())
            window = info.get("training_window", [None, None])
            if isinstance(window, list) and len(window) >= 2 and window[1]:
                last_train_ts = pd.Timestamp(window[1])
            preprocessing = info.get("preprocessing", {})
            if isinstance(preprocessing, dict):
                feature_metadata = dict(preprocessing)
                cols = preprocessing.get("feature_columns")
                if cols:
                    feature_columns = list(cols)
                label_col = preprocessing.get("label_column", label_col)
                tick_val = preprocessing.get("last_tick_ts")
                train_val = preprocessing.get("last_train_ts")
                if tick_val:
                    last_tick_ts = pd.Timestamp(tick_val)
                if train_val:
                    last_train_ts = pd.Timestamp(train_val)
            perf_section = info.get("performance")
            if isinstance(perf_section, dict):
                performance_meta = dict(perf_section)
        except Exception:  # pragma: no cover - best effort
            pass
    if last_tick_ts is None:
        last_tick_ts = last_train_ts
    latest_path = model_dir / "online_latest.joblib"
    state = {
        "model": model,
        "last_tick_ts": last_tick_ts,
        "last_train_ts": last_train_ts,
        "feature_columns": feature_columns,
        "feature_metadata": feature_metadata,
        "label_column": label_col,
    }
    if performance_meta:
        best_thr = performance_meta.get("best_threshold")
        if best_thr is not None:
            try:
                best_val = float(best_thr)
                state["best_threshold"] = best_val
                setattr(model, "best_threshold", best_val)
                setattr(model, "best_threshold_", best_val)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        val_f1 = performance_meta.get("validation_f1")
        if val_f1 is not None:
            try:
                val_score = float(val_f1)
                state["validation_f1"] = val_score
                setattr(model, "validation_f1_", val_score)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        thr_map = _normalise_regime_thresholds(
            performance_meta.get("regime_thresholds")
        )
        if thr_map:
            state["regime_thresholds"] = dict(thr_map)
            setattr(model, "regime_thresholds", dict(thr_map))
            setattr(model, "regime_thresholds_", dict(thr_map))
    joblib.dump(state, latest_path)
    logger.info("Rolled back to model %s", candidate.stem)
    return latest_path


def main() -> None:
    """Command-line entry point for the online training loop."""

    init_logging()
    train_online()


if __name__ == "__main__":
    main()
