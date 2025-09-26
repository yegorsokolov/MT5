"""Generate per-tick probability signals for the EA."""
from mt5.log_utils import setup_logging, log_exceptions, log_predictions
import mt5.log_utils as log_utils

from pathlib import Path
import datetime as _dt
import os
import json
import logging
import importlib
import joblib
import pandas as pd
from mt5.state_manager import (
    load_runtime_state,
    migrate_runtime_state,
    save_runtime_state,
    legacy_runtime_state_exists,
)

import numpy as np

from utils import load_config
from mt5.prediction_cache import PredictionCache
from typing import Any, Mapping
from utils.market_hours import is_market_open
import argparse
from mt5 import backtest
from river import compose
from data.history import load_history_parquet, load_history_config, load_history_iter
from data.features import make_features, make_sequence_arrays
from mt5.train_rl import (
    TradingEnv,
    DiscreteTradingEnv,
    RLLibTradingEnv,
    HierarchicalTradingEnv,
    artifact_dir,
)
import asyncio
from mt5.signal_queue import publish_dataframe_async, get_signal_backend
from models.ensemble import EnsembleModel
from models import model_store
from analysis.concept_drift import ConceptDriftMonitor
from models.conformal import (
    ConformalIntervalParams,
    evaluate_coverage,
    predict_interval,
)

_LOGGING_INITIALIZED = False

_CACHE_ENV_VAR = "MT5_CACHE_DIR"
_CACHE_CFG_KEY = "cache_dir"
_CACHE_SUBDIR = "cache"
_HISTORY_CACHE_FILENAME = "history.parquet"
_MACRO_CACHE_FILENAME = "macro.csv"

_DEFAULT_HISTORY_CHUNK_SIZE = 5000
_DEFAULT_HISTORY_LOOKBACK_DAYS = 3.0
_HISTORY_STREAM_BATCH_KEYS = (
    "history_stream_batch_size",
    "history_stream_chunk_size",
    "history_batch_size",
)
_HISTORY_LOOKBACK_KEYS = (
    "history_max_lookback_days",
    "history_lookback_days",
)


def init_logging() -> logging.Logger:
    """Initialise structured logging for signal generation."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD_KEY = "default"


def _resolve_cache_dir(cfg: Mapping[str, Any] | None = None) -> Path:
    """Return the directory used for local caches.

    Configuration overrides via ``cache_dir`` take precedence, followed by the
    ``MT5_CACHE_DIR`` environment variable.  When no overrides are supplied the
    helper falls back to ``LOG_DIR / "cache"``.
    """

    override: Path | None = None
    if isinstance(cfg, Mapping):
        raw_override = cfg.get(_CACHE_CFG_KEY)
        if raw_override:
            try:
                override = Path(raw_override)
            except TypeError:  # pragma: no cover - defensive conversion
                try:
                    override = Path(str(raw_override))
                except Exception:  # pragma: no cover - defensive conversion
                    override = None
    if override is None:
        env_override = os.getenv(_CACHE_ENV_VAR)
        if env_override:
            try:
                override = Path(env_override)
            except TypeError:  # pragma: no cover - defensive conversion
                override = None

    if override is not None:
        return override.expanduser()

    base_dir = getattr(log_utils, "LOG_DIR", Path(__file__).resolve().parents[1] / "logs")
    try:
        base_path = Path(base_dir)
    except TypeError:  # pragma: no cover - defensive conversion
        base_path = Path(__file__).resolve().parents[1] / "logs"
    return base_path.expanduser() / _CACHE_SUBDIR


def _normalise_thresholds(
    thresholds: dict[int | str, float | int] | None,
) -> tuple[dict[int, float], float | None]:
    """Return ``(per_regime, default)`` thresholds parsed from metadata."""

    if not thresholds:
        return {}, None

    regime_thresholds: dict[int, float] = {}
    default_threshold: float | None = None

    for key, value in thresholds.items():
        if isinstance(key, str) and key.lower() == DEFAULT_THRESHOLD_KEY:
            try:
                default_threshold = float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            continue
        try:
            norm_key = int(key)
        except (TypeError, ValueError):
            continue
        try:
            regime_thresholds[norm_key] = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue

    return regime_thresholds, default_threshold


def _resolve_threshold_metadata(
    models: list[Any], online_model: Any | None, cfg: Mapping[str, Any]
) -> tuple[dict[int, float], float]:
    """Return combined regime thresholds and default cutoff for predictions."""

    base_threshold = float(cfg.get("threshold", 0.5))
    regime_thresholds: dict[int, float] = {}
    best_threshold: float | None = None

    if models:
        primary = models[0]
        regime_thresholds, default_override = _normalise_thresholds(
            getattr(primary, "regime_thresholds", None)
        )
        best_attr = getattr(primary, "best_threshold_", None)
        if best_attr is None:
            best_attr = getattr(primary, "best_threshold", None)
        if best_attr is not None:
            try:
                best_threshold = float(best_attr)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                best_threshold = None
        if best_threshold is None and default_override is not None:
            best_threshold = default_override

    online_best: float | None = None
    online_thresholds: dict[int, float] = {}
    if online_model is not None:
        online_thresholds, online_default = _normalise_thresholds(
            getattr(online_model, "regime_thresholds", None)
        )
        online_attr = getattr(online_model, "best_threshold_", None)
        if online_attr is None:
            online_attr = getattr(online_model, "best_threshold", None)
        if online_attr is not None:
            try:
                online_best = float(online_attr)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                online_best = None
        if online_best is None and online_default is not None:
            online_best = online_default

    if not regime_thresholds and online_thresholds:
        regime_thresholds = dict(online_thresholds)

    if best_threshold is None and online_best is not None:
        best_threshold = online_best

    if best_threshold is None:
        best_threshold = base_threshold

    return regime_thresholds, float(best_threshold)


def _combine_interval_params(
    params: list[ConformalIntervalParams],
) -> tuple[float | dict[int | str, float] | None, float | None]:
    """Aggregate interval parameters across an ensemble."""

    if not params:
        return None, None
    has_mapping = any(isinstance(p.quantiles, Mapping) for p in params)
    quantiles: float | dict[int | str, float] | None
    if has_mapping:
        regimes: set[object] = set()
        for p in params:
            q = p.quantiles
            if isinstance(q, Mapping):
                regimes.update(q.keys())
        combined: dict[int | str, float] = {}
        for reg in regimes:
            values: list[float] = []
            for p in params:
                q = p.quantiles
                if isinstance(q, Mapping):
                    if reg in q:
                        values.append(float(q[reg]))
                else:
                    values.append(float(q))
            if not values:
                continue
            key: int | str | object = reg
            if isinstance(reg, (int, np.integer)):
                key = int(reg)
            else:
                try:
                    key = int(str(reg))
                except (TypeError, ValueError):
                    key = reg
            combined[int(key) if isinstance(key, int) else key] = float(
                np.median(values)
            )
        quantiles = combined or None
    else:
        quantiles = float(np.median([float(p.quantiles) for p in params]))
    coverage_values = [float(p.coverage) for p in params if p.coverage is not None]
    coverage = float(np.mean(coverage_values)) if coverage_values else None
    return quantiles, coverage


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    """Return ``value`` converted to a timezone-naive ``Timestamp``."""

    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:  # pragma: no cover - defensive parsing
        return None
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        try:
            dt_val = ts.to_pydatetime()
        except Exception:  # pragma: no cover - defensive conversion
            return None
        if getattr(dt_val, "tzinfo", None) is not None:
            dt_val = dt_val.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        ts = pd.Timestamp(dt_val)
    return ts


def _resolve_history_streaming_params(
    cfg: Mapping[str, Any] | None,
) -> tuple[int, float | None]:
    """Return ``(chunk_size, lookback_days)`` derived from ``cfg``."""

    chunk_size = _DEFAULT_HISTORY_CHUNK_SIZE
    lookback_days: float | None = _DEFAULT_HISTORY_LOOKBACK_DAYS
    if isinstance(cfg, Mapping):
        for key in _HISTORY_STREAM_BATCH_KEYS:
            raw = cfg.get(key)
            if raw is None:
                continue
            try:
                chunk_size = int(raw)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                continue
            else:
                break
        if chunk_size <= 0:
            chunk_size = _DEFAULT_HISTORY_CHUNK_SIZE

        for key in _HISTORY_LOOKBACK_KEYS:
            raw = cfg.get(key)
            if raw is None:
                continue
            try:
                lookback_days = float(raw)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                continue
            else:
                break
    if lookback_days is not None and lookback_days <= 0:
        lookback_days = None
    return chunk_size, lookback_days


def _load_incremental_history(
    path: Path,
    *,
    last_timestamp: Any,
    chunk_size: int,
    lookback_days: float | None,
) -> pd.DataFrame:
    """Stream and return history rows newer than ``last_timestamp``.

    When ``lookback_days`` is provided, rows preceding ``last_timestamp`` by at
    most that horizon are retained to provide feature context.  If streaming
    fails, the helper falls back to ``load_history_parquet``.
    """

    ts_cutoff = _coerce_timestamp(last_timestamp)
    if ts_cutoff is None:
        logger.warning("Invalid last timestamp %r; loading full history", last_timestamp)
        return load_history_parquet(path)

    if chunk_size <= 0:
        chunk_size = _DEFAULT_HISTORY_CHUNK_SIZE

    earliest: pd.Timestamp | None = None
    if lookback_days is not None:
        try:
            earliest = ts_cutoff - pd.Timedelta(days=float(lookback_days))
        except Exception:  # pragma: no cover - defensive timedelta
            earliest = None

    frames: list[pd.DataFrame] = []
    context_rows = 0
    new_rows = 0
    last_columns: list[str] | None = None

    try:
        for chunk in load_history_iter(path, chunk_size):
            if last_columns is None:
                last_columns = list(chunk.columns)
            if chunk.empty or "Timestamp" not in chunk.columns:
                continue
            ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
            if earliest is not None:
                context_mask = (ts >= earliest) & (ts <= ts_cutoff)
                if bool(context_mask.any()):
                    context = chunk.loc[context_mask].copy()
                    if not context.empty:
                        frames.append(context)
                        context_rows += len(context)
            new_mask = ts > ts_cutoff
            if bool(new_mask.any()):
                newer = chunk.loc[new_mask].copy()
                if not newer.empty:
                    frames.append(newer)
                    new_rows += len(newer)
    except Exception:  # pragma: no cover - best effort logging
        logger.exception(
            "Streaming history from %s failed; falling back to eager load", path
        )
        return load_history_parquet(path)

    if not frames:
        logger.info("No new history rows found after %s", ts_cutoff.isoformat())
        if last_columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=last_columns)

    logger.info(
        "Streamed %d rows (%d context) newer than %s", new_rows, context_rows, ts_cutoff
    )
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _apply_regression_trunk(model: Any, data: pd.DataFrame | np.ndarray):
    """Transform ``data`` using the regression trunk defined on ``model``."""

    trunk = getattr(model, "regression_trunk_", None)
    if trunk is None:
        steps = getattr(model, "steps", None)
        if steps and len(steps) > 1:
            try:
                return model[:-1].transform(data)
            except Exception:  # pragma: no cover - best effort fallback
                logger.exception("Regression trunk transformation failed; using raw features")
        return data

    if isinstance(trunk, str):
        trunk_steps = [trunk]
    elif isinstance(trunk, (list, tuple)):
        trunk_steps = list(trunk)
    else:
        if hasattr(trunk, "transform"):
            try:
                return trunk.transform(data)
            except Exception:  # pragma: no cover - best effort fallback
                logger.exception("Custom regression trunk transform failed")
                return data
        return data

    transformed = data
    named_steps = getattr(model, "named_steps", {})
    for name in trunk_steps:
        transformer = named_steps.get(name)
        if transformer is None:
            logger.debug("Regression trunk step %s missing on model", name)
            continue
        try:
            transformed = transformer.transform(transformed)
        except Exception:  # pragma: no cover - best effort fallback
            logger.exception("Regression trunk step %s failed", name)
            return data
    return transformed


def _get_regression_estimator(model: Any) -> Any | None:
    """Return estimator responsible for regression heads on ``model``."""

    estimator = getattr(model, "regression_estimator_", None)
    if estimator is not None:
        return estimator
    named_steps = getattr(model, "named_steps", None)
    steps = getattr(model, "steps", None)
    if named_steps is not None and steps:
        last_name = steps[-1][0]
        if last_name in named_steps:
            return named_steps[last_name]
    if hasattr(model, "predict_regression"):
        return model
    return None


def compute_regression_estimates(
    models: list[Any], df: pd.DataFrame, fallback_features: list[str]
) -> dict[str, np.ndarray]:
    """Return aggregated regression predictions for ``df`` across ``models``."""

    if df.empty:
        return {}

    results: dict[str, list[np.ndarray]] = {}
    n_samples = len(df)

    for mdl in models:
        heads = getattr(mdl, "regression_heads_", {})
        if not heads:
            continue

        reg_cols = getattr(mdl, "regression_feature_columns_", None)
        if reg_cols:
            columns = [c for c in reg_cols if c in df.columns]
        else:
            columns = [c for c in fallback_features if c in df.columns]
        if not columns:
            logger.debug("No regression features available for model %s", getattr(mdl, "__class__", type(mdl)))
            continue

        X_reg = df[columns]
        shared = _apply_regression_trunk(mdl, X_reg)
        estimator = _get_regression_estimator(mdl)
        if estimator is None or not hasattr(estimator, "predict_regression"):
            logger.debug("Model %s lacks regression estimator", getattr(mdl, "__class__", type(mdl)))
            continue

        for head_name, head_info in heads.items():
            head_type = ""
            if isinstance(head_info, Mapping):
                raw_type = head_info.get("type")
                head_type = str(raw_type).lower() if raw_type is not None else ""
            if head_type and head_type != "multi_task":
                continue
            try:
                preds = estimator.predict_regression(shared, head_name)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Regression prediction for head %s failed", head_name)
                continue
            arr = np.asarray(preds, dtype=float).reshape(-1)
            if arr.size == 0:
                continue
            if len(arr) != n_samples:
                if len(arr) < n_samples:
                    arr = np.pad(arr, (0, n_samples - len(arr)), mode="edge")
                else:
                    arr = arr[:n_samples]
            results.setdefault(head_name, []).append(arr)

    aggregated: dict[str, np.ndarray] = {}
    for head_name, arrays in results.items():
        try:
            aggregated[head_name] = np.mean(np.vstack(arrays), axis=0)
        except ValueError:  # pragma: no cover - guard against malformed arrays
            logger.exception("Failed to aggregate regression outputs for %s", head_name)
    return aggregated


def apply_regime_thresholds(
    probs: np.ndarray,
    regimes: np.ndarray,
    thresholds: dict[int | str, float | int] | None,
    default_threshold: float,
) -> np.ndarray:
    """Return binary predictions using per-regime probability thresholds."""

    if len(probs) == 0:
        return np.zeros(0, dtype=int)

    thr_map, override_default = _normalise_thresholds(thresholds)
    effective_default = (
        default_threshold if override_default is None else float(override_default)
    )
    regimes_arr = np.asarray(regimes)
    if regimes_arr.shape[0] != len(probs):
        raise ValueError("Regimes and probabilities must have the same length")

    preds = np.zeros(len(probs), dtype=int)
    unique_regimes = np.unique(regimes_arr)
    for reg in unique_regimes:
        mask = regimes_arr == reg
        if not np.any(mask):
            continue
        thr = thr_map.get(int(reg), effective_default)
        preds[mask] = (probs[mask] >= thr).astype(int)
    return preds


def _validate_account_id(account_id: str | None) -> str | None:
    """Ensure ``account_id`` contains only digits."""

    if account_id is None:
        return None
    if not str(account_id).isdigit():
        raise ValueError(f"Invalid MT5 account ID: {account_id}")
    return str(account_id)


def load_models(paths, versions=None, return_meta: bool = False):
    """Load multiple models from paths or version identifiers.

    When ``return_meta`` is True the function also returns a meta-classifier
    loaded from the ``meta_model_id`` recorded in the primary model's metadata.
    """

    models = []
    feature_list = None
    meta_model = None
    versions = versions or []
    for vid in versions:
        try:
            m, meta = model_store.load_model(vid)
            perf = meta.get("performance", {})
            thr_map, thr_default = _normalise_thresholds(
                perf.get("regime_thresholds")
            )
            if thr_map or thr_default is not None:
                combined_thresholds: dict[int | str, float] = dict(thr_map)
                if thr_default is not None:
                    combined_thresholds[DEFAULT_THRESHOLD_KEY] = float(thr_default)
                setattr(m, "regime_thresholds", combined_thresholds)
                setattr(m, "regime_thresholds_", combined_thresholds)
            interval_params: ConformalIntervalParams | None = None
            interval_blob = perf.get("interval")
            if interval_blob:
                try:
                    interval_params = ConformalIntervalParams.from_dict(interval_blob)
                except Exception:
                    logger.exception(
                        "Failed to deserialize interval parameters for %s", vid
                    )
            if interval_params is None:
                q = perf.get("interval_q")
                if q is not None:
                    coverage_val = perf.get("interval_coverage")
                    coverage = float(coverage_val) if coverage_val is not None else None
                    coverage_by_regime = perf.get("interval_coverage_by_regime")
                    if isinstance(coverage_by_regime, Mapping):
                        cbr: dict[int | str, float] = {}
                        for key, value in coverage_by_regime.items():
                            try:
                                norm_key = int(key)
                            except (TypeError, ValueError):
                                norm_key = key
                            cbr[norm_key] = float(value)
                        coverage_by_regime = cbr
                    else:
                        coverage_by_regime = None
                    interval_params = ConformalIntervalParams(
                        alpha=float(perf.get("interval_alpha", 0.1)),
                        quantiles=q,
                        coverage=coverage,
                        coverage_by_regime=coverage_by_regime,
                    )
            if interval_params is not None:
                setattr(m, "interval_params", interval_params)
                setattr(m, "interval_q", interval_params.quantiles)
                setattr(m, "interval_alpha", interval_params.alpha)
                setattr(m, "interval_coverage", interval_params.coverage)
                if interval_params.coverage_by_regime:
                    setattr(
                        m,
                        "interval_coverage_by_regime",
                        dict(interval_params.coverage_by_regime),
                    )
            if meta_model is None:
                meta_id = perf.get("meta_model_id")
                if meta_id:
                    try:
                        meta_model, _ = model_store.load_model(meta_id)
                    except FileNotFoundError:
                        meta_model = None
            models.append(m)
            training_cfg = meta.get("training_config", {})
            if hasattr(training_cfg, "model_dump"):
                training_data = training_cfg.model_dump()
            elif isinstance(training_cfg, Mapping):
                training_data = dict(training_cfg)
            else:
                training_data = training_cfg or {}
            model_type_meta = None
            if isinstance(training_data, Mapping):
                training_section = training_data.get("training")
                if isinstance(training_section, Mapping):
                    model_type_meta = training_section.get("model_type")
                if model_type_meta is None:
                    model_type_meta = training_data.get("model_type")
                if model_type_meta is None:
                    model_section = training_data.get("model")
                    if isinstance(model_section, Mapping):
                        raw_type = model_section.get("type")
                        if raw_type == "cross_modal_transformer":
                            model_type_meta = "cross_modal"
            if model_type_meta:
                setattr(m, "model_type_", str(model_type_meta).lower())
            if feature_list is None:
                feature_list = meta.get("features") or meta.get(
                    "training_config", {}
                ).get("features")
        except FileNotFoundError:
            logger.warning("Model version %s not found", vid)
    for p in paths:
        mp = Path(p)
        if not mp.is_absolute():
            mp = Path(__file__).resolve().parent / p
        if not mp.exists():
            continue
        model_obj = joblib.load(mp)
        metadata_path = mp.with_name(f"{mp.stem}_metadata.json")
        metadata_blob: dict[str, object] | None = None
        if metadata_path.exists():
            try:
                with metadata_path.open(encoding="utf-8") as fh:
                    metadata_blob = json.load(fh)
            except Exception:  # pragma: no cover - best effort metadata loading
                logger.exception("Failed to read metadata from %s", metadata_path)
            else:
                perf_section = metadata_blob.get("performance", {})
                if isinstance(perf_section, Mapping):
                    thresholds = perf_section.get("regime_thresholds")
                else:
                    thresholds = None
                if not thresholds and isinstance(metadata_blob, Mapping):
                    thresholds = metadata_blob.get("regime_thresholds")
                norm_thresholds, default_thr = _normalise_thresholds(thresholds)
                if norm_thresholds or default_thr is not None:
                    combined_thresholds: dict[int | str, float] = dict(norm_thresholds)
                    if default_thr is not None:
                        combined_thresholds[DEFAULT_THRESHOLD_KEY] = float(default_thr)
                    setattr(model_obj, "regime_thresholds", combined_thresholds)
                    setattr(model_obj, "regime_thresholds_", combined_thresholds)
                if feature_list is None:
                    feats_blob = metadata_blob.get("features")
                    if isinstance(feats_blob, list):
                        feature_list = feats_blob
                    else:
                        feats_blob = metadata_blob.get("training_features")
                        if isinstance(feats_blob, list):
                            feature_list = feats_blob
                setattr(model_obj, "model_metadata", metadata_blob)
        models.append(model_obj)
    if return_meta:
        return models, feature_list, meta_model
    return models, feature_list


def bayesian_average(prob_arrays):
    """Combine probabilities using a simple Bayesian model averaging."""
    logits = [np.log(p / (1 - p + 1e-12)) for p in prob_arrays]
    avg_logit = np.mean(logits, axis=0)
    return 1 / (1 + np.exp(-avg_logit))


def meta_transformer_signals(df, features, cfg):
    """Return probabilities from the multi-head transformer if available."""
    if not cfg.get("use_meta_model", False):
        return np.zeros(len(df))
    try:  # optional torch dependency
        import torch
    except Exception:
        return np.zeros(len(df))
    from models.multi_head import MultiHeadTransformer
    from utils.resource_monitor import monitor

    artifact_root = cfg.get("artifact_dir") if isinstance(cfg, Mapping) else None
    if artifact_root:
        model_dir = Path(artifact_root)
    else:
        base_dir = getattr(log_utils, "LOG_DIR", Path(__file__).resolve().parents[1] / "logs")
        model_dir = Path(base_dir) / "nn_artifacts"
    model_path = model_dir / "model_transformer.pt"
    if not model_path.exists():
        return np.zeros(len(df))

    seq_len = cfg.get("sequence_length", 50)
    feat = [f for f in features if f != "SymbolCode"]
    X, _ = make_sequence_arrays(df, feat, seq_len)
    if len(X) == 0:
        return np.zeros(len(df))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_symbols = int(df["Symbol"].nunique()) if "Symbol" in df.columns else 1
    num_regimes = (
        int(df["market_regime"].nunique()) if "market_regime" in df.columns else None
    )
    model = MultiHeadTransformer(
        len(feat),
        num_symbols=num_symbols,
        num_regimes=num_regimes,
        dropout=0.0,
        layer_norm=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    tier = monitor.capabilities.capability_tier()
    TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}
    sym_code = (
        int(df.get("SymbolCode", pd.Series([0])).iloc[0])
        if "SymbolCode" in df.columns
        else 0
    )
    if TIERS.get(tier, 0) < TIERS["gpu"]:
        model.prune_heads([sym_code])
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i : i + 256], dtype=torch.float32).to(device)
            preds.append(model(xb, sym_code).cpu().numpy())
    probs = np.concatenate(preds)
    if len(probs) < len(df):
        probs = np.pad(probs, (0, len(df) - len(probs)), "edge")
    return probs


def _require_module_attr(module_name: str, attr: str, algo: str):
    """Return ``attr`` from ``module_name`` with informative fallback errors."""

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RL algorithm '%s' requires optional dependency '%s'."
            " Install the 'mt5[rl]' extras or provide the dependency manually."
            % (algo, module_name)
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError(
            "Importing optional dependency '%s' for RL algorithm '%s' failed"
            % (module_name, algo)
        ) from exc

    try:
        value = getattr(module, attr)
    except AttributeError as exc:
        raise RuntimeError(
            "Optional dependency '%s' does not expose attribute '%s' required"
            " for RL algorithm '%s'." % (module_name, attr, algo)
        ) from exc

    if value is None:
        raise RuntimeError(
            "Optional dependency '%s' does not provide implementation for '%s'"
            " required for RL algorithm '%s'." % (module_name, attr, algo)
        )

    return value


def _require_stable_baselines_class(attr: str, algo: str):
    """Return a class from ``stable_baselines3`` for ``algo`` with fallbacks."""

    return _require_module_attr("stable_baselines3", attr, algo)


def _require_sb3_contrib_class(attr: str, algo: str):
    """Return a class from ``sb3_contrib`` for ``algo`` with fallbacks."""

    return _require_module_attr("sb3_contrib", attr, algo)


def _require_sb3_qrdqn(algo: str):
    """Return the QRDQN class from ``sb3_contrib.qrdqn`` with fallbacks."""

    return _require_module_attr("sb3_contrib.qrdqn", "QRDQN", algo)


def rl_signals(df, features, cfg):
    """Return probability-like signals from a trained RL agent."""
    artifact_root = artifact_dir(cfg)
    models_dir = artifact_root / "models"
    model_path = models_dir / "model_rl.zip"
    model_rllib = models_dir / "model_rllib"
    model_recurrent = models_dir / "recurrent_rl" / "recurrent_model.zip"
    model_hierarchical = models_dir / "model_hierarchical.zip"
    algo = cfg.get("rl_algorithm", "PPO").upper()
    if algo == "RLLIB":
        if not model_rllib.exists():
            return np.zeros(len(df))
    elif algo == "RECURRENTPPO":
        if not model_recurrent.exists():
            return np.zeros(len(df))
    elif algo == "HIERARCHICALPPO":
        if not model_hierarchical.exists():
            return np.zeros(len(df))
    else:
        if not model_path.exists():
            return np.zeros(len(df))

    rllib_algo = cfg.get("rllib_algorithm", "PPO").upper()
    algo_key = algo
    model = None
    env = None
    try:
        if algo == "PPO":
            PPO = _require_stable_baselines_class("PPO", algo)
            env = TradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            model = PPO.load(model_path, env=env)
        elif algo == "A2C" or algo == "A3C":
            A2C = _require_stable_baselines_class("A2C", algo)
            env = TradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            model = A2C.load(model_path, env=env)
        elif algo == "SAC":
            SAC = _require_stable_baselines_class("SAC", algo)
            env = TradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            model = SAC.load(model_path, env=env)
        elif algo == "TRPO":
            TRPO = _require_sb3_contrib_class("TRPO", algo)
            env = TradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            model = TRPO.load(model_path, env=env)
        elif algo == "RECURRENTPPO":
            RecurrentPPO = _require_sb3_contrib_class("RecurrentPPO", algo)
            env = TradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            model = RecurrentPPO.load(model_recurrent, env=env)
        elif algo == "HIERARCHICALPPO":
            HierarchicalPPO = _require_sb3_contrib_class("HierarchicalPPO", algo)
            env = HierarchicalTradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            model = HierarchicalPPO.load(model_hierarchical, env=env)
        elif algo == "QRDQN":
            QRDQN = _require_sb3_qrdqn(algo)
            env = DiscreteTradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            model = QRDQN.load(model_path, env=env)
        elif algo == "RLLIB":
            try:
                import ray
                from ray.rllib.algorithms.ppo import PPO as RLlibPPO
                from ray.rllib.algorithms.ddpg import DDPG
            except Exception:
                return np.zeros(len(df))

            ray.init(ignore_reinit_error=True, include_dashboard=False)
            env = RLLibTradingEnv(
                df,
                features,
                max_position=cfg.get("rl_max_position", 1.0),
                transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
                risk_penalty=cfg.get("rl_risk_penalty", 0.1),
                var_window=cfg.get("rl_var_window", 30),
                cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
                cvar_window=cfg.get("rl_cvar_window", 30),
            )
            if rllib_algo == "DDPG":
                model = DDPG.from_checkpoint(model_rllib)
            else:
                model = RLlibPPO.from_checkpoint(model_rllib)
        # other algorithms fall through to the default QRDQN fallback below
    except RuntimeError as exc:
        logger.error("RL signal generation unavailable: %s", exc)
        return np.zeros(len(df))
    except Exception:  # pragma: no cover - unexpected RL failures
        logger.exception("Unexpected error during RL signal generation")
        return np.zeros(len(df))
    if model is None or env is None:
        # Default fallback mirrors QRDQN behaviour for unknown algorithms
        try:
            QRDQN = _require_sb3_qrdqn("QRDQN")
        except RuntimeError as exc:
            logger.error("RL signal generation unavailable: %s", exc)
            return np.zeros(len(df))
        env = DiscreteTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            cvar_penalty=cfg.get("rl_cvar_penalty", 0.0),
            cvar_window=cfg.get("rl_cvar_window", 30),
        )
        model = QRDQN.load(model_path, env=env)
        algo_key = "QRDQN"

    if algo_key == "RLLIB":
        obs, _ = env.reset()
    else:
        obs = env.reset()
    done = False
    actions = []
    state = None
    episode_start = np.ones((1,), dtype=bool)
    while not done:
        if algo_key == "RLLIB":
            action = model.compute_single_action(obs)
        elif algo_key == "RECURRENTPPO":
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=True
            )
        elif algo_key == "HIERARCHICALPPO":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        if algo_key == "HIERARCHICALPPO":
            a = float(action["manager"])  # direction for signal
            env_action = action
        else:
            a = float(action[0]) if not np.isscalar(action) else float(action)
            env_action = action
        actions.append(a)
        if algo_key == "RLLIB":
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            obs, _, done, _ = env.step(env_action)
        episode_start = np.array([done], dtype=bool)

    probs = (np.array(actions) > 0).astype(float)
    if len(probs) < len(df):
        probs = np.pad(probs, (0, len(df) - len(probs)), "edge")
    if algo_key == "RLLIB":
        ray.shutdown()
    return probs


@log_exceptions
def main():
    init_logging()
    parser = argparse.ArgumentParser(description="Generate probability signals")
    parser.add_argument(
        "--simulate-closed-market",
        action="store_true",
        help="Force closed market behaviour for testing",
    )
    args = parser.parse_args()

    cfg = load_config()
    cache_dir = _resolve_cache_dir(cfg)
    account_id = _validate_account_id(
        os.getenv("MT5_ACCOUNT_ID") or cfg.get("account_id")
    )
    cache = PredictionCache(
        cfg.get("pred_cache_size", 256), cfg.get("pred_cache_policy", "lru")
    )
    monitor = ConceptDriftMonitor(
        method=cfg.get("drift_method", "adwin"),
        delta=float(cfg.get("drift_delta", 0.002)),
    )

    history_chunk_size, history_lookback_days = _resolve_history_streaming_params(cfg)

    # Reload previous runtime state if available
    state = load_runtime_state(account_id=account_id)
    if account_id and state is None and legacy_runtime_state_exists():
        try:
            migrated_path = migrate_runtime_state(account_id)
        except FileNotFoundError:
            logger.warning(
                "Legacy runtime state was detected but the file disappeared before migration"
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Automatic runtime state migration failed")
        else:
            logger.info("Migrated runtime state to %s", migrated_path)
            state = load_runtime_state(account_id=account_id)
    last_ts_raw = None
    last_ts = None
    prev_models: list[str] = []
    if state:
        last_ts_raw = state.get("last_timestamp")
        last_ts = _coerce_timestamp(last_ts_raw)
        prev_models = state.get("model_versions", [])

    if args.simulate_closed_market or not is_market_open():
        logger.info("Market closed - running backtest and using historical data")
        backtest.run_rolling_backtest(cfg)

    requested_model_type = str(cfg.get("model_type", "lgbm")).lower()
    if requested_model_type == "autogluon":
        logger.warning(
            "The 'autogluon' model_type is deprecated. Using the FLAML-based 'automl' backend instead."
        )
    model_type = "automl" if requested_model_type == "autogluon" else requested_model_type

    configured_paths = cfg.get("ensemble_models")
    if configured_paths:
        model_paths = configured_paths
    elif model_type in {"tabular", "automl"}:
        default_paths = [f"models/{model_type}/model.joblib"]
        if requested_model_type == "autogluon":
            default_paths.insert(0, "models/autogluon/model.joblib")
        model_paths = default_paths
    else:
        model_paths = ["model.joblib"]
    model_versions = cfg.get("model_versions", [])
    env_version = os.getenv("MODEL_VERSION_ID")
    if env_version:
        model_versions.append(env_version)
    models, stored_features, meta_clf = load_models(
        model_paths, model_versions, return_meta=True
    )

    if not models:
        default_relative = (
            f"models/{model_type}/model.joblib"
            if model_type in {"tabular", "automl"}
            else "model.joblib"
        )
        fallback_path = Path(__file__).resolve().parent / default_relative
        if fallback_path.exists():
            models = [joblib.load(fallback_path)]
            if stored_features is None:
                features_path = fallback_path.with_name("features.json")
                if features_path.exists():
                    try:
                        stored_features = json.loads(
                            features_path.read_text(encoding="utf-8")
                        )
                    except Exception:  # pragma: no cover - best effort logging
                        logger.exception(
                            "Failed to read stored features from %s", features_path
                        )
                if stored_features is None:
                    metadata_path = fallback_path.with_name("model_metadata.json")
                    if metadata_path.exists():
                        try:
                            metadata_blob = json.loads(
                                metadata_path.read_text(encoding="utf-8")
                            )
                        except Exception:  # pragma: no cover - defensive guard
                            logger.exception(
                                "Failed to parse model metadata from %s", metadata_path
                            )
                        else:
                            feats = metadata_blob.get("features") or metadata_blob.get(
                                "training_features"
                            )
                            if isinstance(feats, list):
                                stored_features = feats
        elif model_type in {"tabular", "automl"}:
            train_cmd = (
                "python -m mt5.train_automl"
                if model_type == "automl"
                else "python -m mt5.train_tabular"
            )
            raise RuntimeError(
                f"No {model_type} model is available. Train one with '{train_cmd}' "
                "before generating signals."
            )

    # Replay past trades through any newly enabled model versions
    new_versions = [v for v in model_versions if v not in prev_models]
    if new_versions:
        try:
            from analysis.replay_trades import replay_trades

            replay_trades(new_versions)
        except Exception:
            logger.exception("Trade replay for new models failed")

    online_model = None
    online_metadata: Mapping[str, Any] | None = None
    online_path = Path(__file__).resolve().parent / "models" / "online.joblib"
    if cfg.get("use_online_model", False) and online_path.exists():
        try:
            payload = joblib.load(online_path)
            online_metadata = None
            if isinstance(payload, tuple) and len(payload) >= 1:
                online_model = payload[0]
                if len(payload) >= 2 and isinstance(payload[1], Mapping):
                    online_metadata = payload[1]
            elif isinstance(payload, dict):
                online_model = payload.get("model") or payload.get("pipeline")
                meta_candidate = (
                    payload.get("metadata")
                    or payload.get("meta")
                    or payload.get("info")
                )
                if isinstance(meta_candidate, Mapping):
                    online_metadata = meta_candidate
                if online_model is not None:
                    best_val = payload.get("best_threshold")
                    if best_val is not None:
                        try:
                            best_float = float(best_val)
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            best_float = None
                        else:
                            setattr(online_model, "best_threshold", best_float)
                            setattr(online_model, "best_threshold_", best_float)
                    val_f1 = payload.get("validation_f1")
                    if val_f1 is not None:
                        try:
                            setattr(online_model, "validation_f1_", float(val_f1))
                        except (TypeError, ValueError):  # pragma: no cover
                            pass
                    thr_map, thr_default = _normalise_thresholds(
                        payload.get("regime_thresholds")
                    )
                    if thr_map or thr_default is not None:
                        combined_thresholds: dict[int | str, float] = dict(thr_map)
                        if thr_default is not None:
                            combined_thresholds[DEFAULT_THRESHOLD_KEY] = float(thr_default)
                        setattr(online_model, "regime_thresholds", combined_thresholds)
                        setattr(online_model, "regime_thresholds_", combined_thresholds)
            else:
                online_model = payload

            if online_model is not None and isinstance(online_metadata, Mapping):
                perf_section = online_metadata.get("performance")
                if isinstance(perf_section, Mapping):
                    best_val = perf_section.get("best_threshold")
                    if best_val is not None:
                        try:
                            best_float = float(best_val)
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            best_float = None
                        else:
                            setattr(online_model, "best_threshold", best_float)
                            setattr(online_model, "best_threshold_", best_float)
                    val_f1 = perf_section.get("validation_f1")
                    if val_f1 is not None:
                        try:
                            setattr(online_model, "validation_f1_", float(val_f1))
                        except (TypeError, ValueError):  # pragma: no cover
                            pass
                    thr_map, thr_default = _normalise_thresholds(
                        perf_section.get("regime_thresholds")
                    )
                    if thr_map or thr_default is not None:
                        combined_thresholds: dict[int | str, float] = dict(thr_map)
                        if thr_default is not None:
                            combined_thresholds[DEFAULT_THRESHOLD_KEY] = float(thr_default)
                        setattr(online_model, "regime_thresholds", combined_thresholds)
                        setattr(online_model, "regime_thresholds_", combined_thresholds)
            logger.info("Loaded online model from %s", online_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load online model: %s", exc)

    regime_thresholds, default_threshold = _resolve_threshold_metadata(
        models, online_model, cfg
    )
    hist_path_pq = cache_dir / _HISTORY_CACHE_FILENAME
    symbols = cfg.get("symbols") or (
        [cfg.get("symbol")] if cfg.get("symbol") else None
    )
    if hist_path_pq.exists():
        if last_ts is not None or last_ts_raw is not None:
            ts_arg = last_ts if last_ts is not None else last_ts_raw
            df = _load_incremental_history(
                hist_path_pq,
                last_timestamp=ts_arg,
                chunk_size=history_chunk_size,
                lookback_days=history_lookback_days,
            )
        else:
            df = load_history_parquet(hist_path_pq)
    else:
        cfg_root = Path(__file__).resolve().parent
        sym = cfg.get("symbol")
        df = load_history_config(sym, cfg, cfg_root)
        hist_path_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(hist_path_pq, index=False)
    if isinstance(symbols, str):
        symbols = [symbols]
    if symbols and "Symbol" in df.columns:
        df = df[df["Symbol"].isin(symbols)]

    df = make_features(df)

    if last_ts is not None and "Timestamp" in df.columns:
        try:
            ts_series = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df[ts_series > last_ts]
        except Exception:  # pragma: no cover - defensive filter
            logger.debug("Failed to filter rows newer than last timestamp", exc_info=True)
        df = df.reset_index(drop=True)

    # optional macro indicators merged on timestamp
    macro_path = cache_dir / _MACRO_CACHE_FILENAME
    if macro_path.exists():
        macro = pd.read_csv(macro_path)
        macro["Timestamp"] = pd.to_datetime(macro["Timestamp"])
        df = df.merge(macro, on="Timestamp", how="left")
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    if stored_features:
        features = stored_features
    else:
        features = [
            "return",
            "ma_5",
            "ma_10",
            "ma_30",
            "ma_60",
            "ma_h4",
            "volatility_30",
            "spread",
            "rsi_14",
            "hour_sin",
            "hour_cos",
            "news_sentiment",
        ]
        for col in [
            "atr_14",
            "atr_stop_long",
            "atr_stop_short",
            "donchian_high",
            "donchian_low",
            "donchian_break",
        ]:
            if col in df.columns:
                features.append(col)
        features += [
            c
            for c in df.columns
            if c.startswith("cross_corr_")
            or c.startswith("factor_")
            or c.startswith("cross_mom_")
        ]
        if "volume_ratio" in df.columns:
            features.extend(["volume_ratio", "volume_imbalance"])
        if "SymbolCode" in df.columns:
            features.append("SymbolCode")

    if model_type in {"tabular", "automl"}:
        prob_models = [mdl for mdl in models if hasattr(mdl, "predict_proba")]
        if not prob_models:
            train_cmd = (
                "python -m mt5.train_automl"
                if model_type == "automl"
                else "python -m mt5.train_tabular"
            )
            raise RuntimeError(
                f"{model_type.capitalize()} model loading failed – ensure '{train_cmd}' "
                "has been executed successfully."
            )

        def _predict(data: pd.DataFrame) -> np.ndarray:
            preds: list[np.ndarray] = []
            for mdl in prob_models:
                try:
                    output = mdl.predict_proba(data[features])
                except Exception:  # pragma: no cover - best effort logging
                    logger.exception("Tabular-style model prediction failed")
                    continue
                arr = np.asarray(output)
                if arr.ndim == 2:
                    arr = arr[:, -1]
                preds.append(arr)
            if not preds:
                return np.zeros(len(data))
            return np.mean(preds, axis=0)

    else:
        base_models: dict[str, Any] = {}
        if models:
            cross_modal_models = [
                mdl
                for mdl in models
                if getattr(mdl, "model_type_", "").lower() == "cross_modal"
            ]
            gbm_models = [
                mdl
                for mdl in models
                if getattr(mdl, "model_type_", "").lower() != "cross_modal"
            ]

            if gbm_models:

                def _gbm_predict(data: pd.DataFrame) -> np.ndarray:
                    preds: list[np.ndarray] = []
                    for mdl in gbm_models:
                        try:
                            out = mdl.predict_proba(data[features])
                        except Exception:  # pragma: no cover - best effort logging
                            logger.exception("Gradient boosting prediction failed")
                            continue
                        arr = np.asarray(out)
                        if arr.ndim == 2:
                            arr = arr[:, -1]
                        preds.append(arr)
                    if not preds:
                        return np.zeros(len(data))
                    return np.mean(preds, axis=0)

                base_models["lightgbm"] = _gbm_predict

            if cross_modal_models:

                def _cross_modal_predict(data: pd.DataFrame) -> np.ndarray:
                    preds: list[np.ndarray] = []
                    for mdl in cross_modal_models:
                        try:
                            out = mdl.predict_proba(data[features])
                        except Exception:  # pragma: no cover - best effort logging
                            logger.exception("Cross-modal prediction failed")
                            continue
                        arr = np.asarray(out)
                        if arr.ndim == 2:
                            arr = arr[:, -1]
                        preds.append(arr)
                    if not preds:
                        return np.zeros(len(data))
                    return np.mean(preds, axis=0)

                base_models["cross_modal"] = _cross_modal_predict

        if cfg.get("use_meta_model", False):
            base_models["transformer"] = lambda d: meta_transformer_signals(
                d, features, cfg
            )

        if online_model is not None:

            def _online_predict(data: pd.DataFrame) -> np.ndarray:
                return np.array(
                    [
                        online_model.predict_proba_one(row).get(1, 0.0)
                        for row in data[features].to_dict("records")
                    ]
                )

            base_models["online"] = _online_predict

        if cfg.get("blend_with_rl", False):
            base_models["rl"] = lambda d: rl_signals(d, features, cfg)

        ensemble = EnsembleModel(base_models) if base_models else None

        def _predict(data: pd.DataFrame) -> np.ndarray:
            if ensemble is None:
                return np.zeros(len(data))
            return ensemble.predict(data)["ensemble"]

    regression_outputs = compute_regression_estimates(models, df, features)
    expected_returns = np.asarray(
        regression_outputs.get("abs_return", np.zeros(len(df))), dtype=float
    )
    predicted_volatility = np.asarray(
        regression_outputs.get("volatility", np.zeros(len(df))), dtype=float
    )
    if len(expected_returns) != len(df):
        expected_returns = np.resize(expected_returns, len(df))
    if len(predicted_volatility) != len(df):
        predicted_volatility = np.resize(predicted_volatility, len(df))
    expected_returns = np.nan_to_num(expected_returns, nan=0.0)
    predicted_volatility = np.nan_to_num(predicted_volatility, nan=0.0)

    hashes = pd.util.hash_pandas_object(df[features], index=False).values
    probs = np.zeros(len(df))
    pred_dict = {"ensemble": probs}
    miss_idx: list[int] = []
    for i, h in enumerate(hashes):
        val = cache.get(int(h))
        if isinstance(val, Mapping):
            prob_val = val.get("prob")
            exp_val = val.get("expected_return")
            vol_val = val.get("predicted_volatility") or val.get("volatility")
            if prob_val is not None:
                probs[i] = float(prob_val)
            else:
                miss_idx.append(i)
            if exp_val is not None:
                expected_returns[i] = float(exp_val)
            if vol_val is not None:
                predicted_volatility[i] = float(vol_val)
            if prob_val is None:
                continue
        elif val is not None:
            miss_idx.append(i)
        else:
            miss_idx.append(i)
    if miss_idx:
        sub_df = df.iloc[miss_idx]
        new_probs = _predict(sub_df)
        for j, idx in enumerate(miss_idx):
            prob = float(new_probs[j])
            probs[idx] = prob
            monitor.update(sub_df.iloc[j][features], prob)

    if cache.maxsize > 0 and len(hashes):
        for i, h in enumerate(hashes):
            cache.set(
                int(h),
                {
                    "prob": float(probs[i]),
                    "expected_return": float(expected_returns[i]),
                    "predicted_volatility": float(predicted_volatility[i]),
                },
            )

    ma_ok = df["ma_cross"] == 1
    rsi_ok = df["rsi_14"] > cfg.get("rsi_buy", 55)

    boll_ok = True
    if "boll_break" in df.columns:
        boll_ok = df["boll_break"] == 1

    vol_ok = True
    if "volume_spike" in df.columns:
        vol_ok = df["volume_spike"] == 1

    macro_ok = True
    if "macro_indicator" in df.columns:
        macro_ok = df["macro_indicator"] > cfg.get("macro_threshold", 0.0)

    news_ok = True
    if not cfg.get("enable_news_trading", True):
        window = cfg.get("avoid_news_minutes", 5)
        if "nearest_news_minutes" in df.columns:
            news_ok = df["nearest_news_minutes"] > window

    sent_ok = True
    if "news_sentiment" in df.columns:
        sent_ok = df["news_sentiment"] > 0

    mom_ok = True
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    if factor_cols:
        mom_ok = df[factor_cols[0]] > 0

    exp_ok = np.ones(len(df), dtype=bool)
    min_expected = cfg.get("min_expected_return")
    if min_expected is not None:
        threshold_exp = float(min_expected)
        exp_ok = expected_returns >= threshold_exp

    pred_vol_ok = np.ones(len(df), dtype=bool)
    max_pred_vol = cfg.get("max_predicted_volatility")
    if max_pred_vol is not None:
        vol_threshold = float(max_pred_vol)
        pred_vol_ok = predicted_volatility <= vol_threshold

    combined = np.where(
        ma_ok
        & rsi_ok
        & boll_ok
        & vol_ok
        & macro_ok
        & news_ok
        & sent_ok
        & mom_ok
        & exp_ok
        & pred_vol_ok,
        probs,
        0.0,
    )
    if meta_clf is not None:
        meta_feat = pd.DataFrame(
            {
                "prob": combined,
                "confidence": np.abs(combined - 0.5) * 2,
            }
        )
        keep = meta_clf.predict(meta_feat) == 1
        combined = np.where(keep, combined, 0.0)

    regimes = df["market_regime"] if "market_regime" in df.columns else None
    interval_params_list = [
        getattr(m, "interval_params", None)
        for m in models
        if getattr(m, "interval_params", None) is not None
    ]
    interval_params_list = [
        p for p in interval_params_list if isinstance(p, ConformalIntervalParams)
    ]
    combined_q, avg_cov = _combine_interval_params(interval_params_list)
    if combined_q is None:
        legacy_qs = [
            getattr(m, "interval_q", None)
            for m in models
            if getattr(m, "interval_q", None) is not None
        ]
        values: list[float] = []
        for q in legacy_qs:
            if isinstance(q, Mapping):
                vals = list(q.values())
                if vals:
                    values.append(float(np.median(vals)))
            elif q is not None:
                values.append(float(q))
        if values:
            combined_q = float(np.median(values))
    if avg_cov is None:
        coverage_values = [
            getattr(m, "interval_coverage", None)
            for m in models
            if getattr(m, "interval_coverage", None) is not None
        ]
        if coverage_values:
            avg_cov = float(np.mean([float(c) for c in coverage_values]))
    coverage_target = cfg.get("interval_coverage_target")
    if coverage_target is None:
        coverage_target = cfg.get("interval_min_coverage")
    coverage_target = float(coverage_target) if coverage_target is not None else None
    if coverage_target is not None and avg_cov is not None:
        if avg_cov < coverage_target:
            logger.warning(
                "Interval coverage %.3f below target %.3f; suppressing signals",
                avg_cov,
                coverage_target,
            )
            combined = np.zeros_like(combined)
        else:
            logger.info(
                "Interval coverage %.3f meets target %.3f",
                avg_cov,
                coverage_target,
            )
    elif avg_cov is not None:
        logger.info("Interval coverage from calibration: %.3f", avg_cov)

    regimes_arr = None
    if regimes is not None:
        regimes_arr = (
            regimes.to_numpy() if hasattr(regimes, "to_numpy") else np.asarray(regimes)
        )
    default_thr = float(default_threshold)
    if regimes_arr is not None and regime_thresholds:
        preds = apply_regime_thresholds(
            combined,
            regimes_arr,
            regime_thresholds,
            default_thr,
        )
    else:
        preds = (combined >= default_thr).astype(int)

    out = pd.DataFrame(
        {
            "Timestamp": df["Timestamp"],
            "Symbol": cfg.get("symbol"),
            "prob": combined,
            "pred": preds,
            "expected_return": expected_returns,
            "predicted_volatility": predicted_volatility,
        }
    )
    if avg_cov is not None:
        out["interval_avg_coverage"] = avg_cov
    if coverage_target is not None:
        out["interval_coverage_target"] = coverage_target
    quantiles_to_use = combined_q
    regimes_for_interval = None
    if isinstance(quantiles_to_use, Mapping):
        regimes_for_interval = regimes_arr
        if regimes_for_interval is None:
            vals = list(quantiles_to_use.values())
            quantiles_to_use = float(np.median(vals)) if vals else None
            regimes_for_interval = None
    if quantiles_to_use is not None:
        lower, upper = predict_interval(
            combined,
            quantiles_to_use,
            regimes_for_interval,
        )
        out["ci_lower"] = lower
        out["ci_upper"] = upper
        y_true = None
        if "y_true" in df.columns:
            y_true = df["y_true"].to_numpy()
        elif "label" in df.columns:
            y_true = df["label"].to_numpy()
        if y_true is not None:
            out["interval_covered"] = ((y_true >= lower) & (y_true <= upper)).astype(int)
            cov = evaluate_coverage(y_true, lower, upper)
            logger.info("Interval coverage: %.3f", cov)
    log_df = df[["Timestamp"] + features].copy()
    log_df["Symbol"] = cfg.get("symbol")
    for name, arr in pred_dict.items():
        log_df[f"prob_{name}"] = arr
    log_df["prob"] = combined
    log_df["pred"] = preds
    log_df["expected_return"] = expected_returns
    log_df["predicted_volatility"] = predicted_volatility
    log_predictions(log_df)
    fmt = os.getenv("SIGNAL_FORMAT", "protobuf")
    queue = get_signal_backend(cfg)
    if queue is not None:
        asyncio.run(publish_dataframe_async(queue, out, fmt=fmt))
        logger.info("Signals published")

    # Persist runtime state for recovery on next startup
    try:
        from data.trade_log import TradeLog

        open_positions = []
        tl_path = Path("/var/lib/mt5bot/trades.db")
        if tl_path.exists():
            open_positions = TradeLog(tl_path).get_open_positions()
    except Exception:
        open_positions = []

    try:
        last_processed = (
            pd.to_datetime(df["Timestamp"]).max().isoformat() if not df.empty else ""
        )
        save_runtime_state(
            last_processed, open_positions, model_versions, account_id=account_id
        )
    except Exception:
        logger.exception("Failed to persist runtime state")


if __name__ == "__main__":
    main()
