"""Generate per-tick probability signals for the EA."""

from log_utils import setup_logging, log_exceptions, log_predictions

from pathlib import Path
import os
import json
import logging
import joblib
import pandas as pd
from state_manager import (
    load_runtime_state,
    migrate_runtime_state,
    save_runtime_state,
    legacy_runtime_state_exists,
)

import numpy as np

from utils import load_config
from prediction_cache import PredictionCache
from typing import Any, Mapping
from utils.market_hours import is_market_open
import argparse
import backtest
from river import compose
from data.history import load_history_parquet, load_history_config
from data.features import make_features, make_sequence_arrays
from train_rl import (
    TradingEnv,
    DiscreteTradingEnv,
    RLLibTradingEnv,
    HierarchicalTradingEnv,
)
from stable_baselines3 import PPO, SAC, A2C
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib import TRPO, RecurrentPPO

try:  # optional import for hierarchical PPO
    from sb3_contrib import HierarchicalPPO  # type: ignore
except Exception:  # pragma: no cover - algorithm may not be available
    HierarchicalPPO = None  # type: ignore
import asyncio
from signal_queue import publish_dataframe_async, get_signal_backend
from models.ensemble import EnsembleModel
from models import model_store
from analysis.concept_drift import ConceptDriftMonitor
from models.conformal import (
    ConformalIntervalParams,
    evaluate_coverage,
    predict_interval,
)

setup_logging()
logger = logging.getLogger(__name__)


def _normalise_thresholds(
    thresholds: dict[int | str, float | int] | None,
) -> dict[int, float]:
    """Cast regime thresholds to ``int -> float`` mapping."""

    if not thresholds:
        return {}
    return {int(k): float(v) for k, v in thresholds.items()}


def _resolve_threshold_metadata(
    models: list[Any], online_model: Any | None, cfg: Mapping[str, Any]
) -> tuple[dict[int, float], float]:
    """Return combined regime thresholds and default cutoff for predictions."""

    base_threshold = float(cfg.get("threshold", 0.5))
    regime_thresholds: dict[int, float] = {}
    best_threshold: float | None = None

    if models:
        primary = models[0]
        regime_thresholds = _normalise_thresholds(
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

    online_best: float | None = None
    online_thresholds: dict[int, float] = {}
    if online_model is not None:
        online_thresholds = _normalise_thresholds(
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

    thr_map = _normalise_thresholds(thresholds)
    regimes_arr = np.asarray(regimes)
    if regimes_arr.shape[0] != len(probs):
        raise ValueError("Regimes and probabilities must have the same length")

    preds = np.zeros(len(probs), dtype=int)
    unique_regimes = np.unique(regimes_arr)
    for reg in unique_regimes:
        mask = regimes_arr == reg
        if not np.any(mask):
            continue
        thr = thr_map.get(int(reg), default_threshold)
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
            thr = _normalise_thresholds(perf.get("regime_thresholds"))
            if thr:
                setattr(m, "regime_thresholds", thr)
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
                norm_thresholds = _normalise_thresholds(thresholds)
                if norm_thresholds:
                    setattr(model_obj, "regime_thresholds", norm_thresholds)
                    setattr(model_obj, "regime_thresholds_", norm_thresholds)
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

    model_path = Path(__file__).resolve().parent / "model_transformer.pt"
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


def rl_signals(df, features, cfg):
    """Return probability-like signals from a trained RL agent."""
    model_path = Path(__file__).resolve().parent / "model_rl.zip"
    model_rllib = Path(__file__).resolve().parent / "model_rllib"
    model_recurrent = (
        Path(__file__).resolve().parent
        / "models"
        / "recurrent_rl"
        / "recurrent_model.zip"
    )
    model_hierarchical = Path(__file__).resolve().parent / "model_hierarchical.zip"
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
    if algo == "PPO":
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
        if HierarchicalPPO is None:
            return np.zeros(len(df))
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
    else:
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

    if algo == "RLLIB":
        obs, _ = env.reset()
    else:
        obs = env.reset()
    done = False
    actions = []
    state = None
    episode_start = np.ones((1,), dtype=bool)
    while not done:
        if algo == "RLLIB":
            action = model.compute_single_action(obs)
        elif algo == "RECURRENTPPO":
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=True
            )
        elif algo == "HIERARCHICALPPO":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        if algo == "HIERARCHICALPPO":
            a = float(action["manager"])  # direction for signal
            env_action = action
        else:
            a = float(action[0]) if not np.isscalar(action) else float(action)
            env_action = action
        actions.append(a)
        if algo == "RLLIB":
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            obs, _, done, _ = env.step(env_action)
        episode_start = np.array([done], dtype=bool)

    probs = (np.array(actions) > 0).astype(float)
    if len(probs) < len(df):
        probs = np.pad(probs, (0, len(df) - len(probs)), "edge")
    if algo == "RLLIB":
        ray.shutdown()
    return probs


@log_exceptions
def main():
    parser = argparse.ArgumentParser(description="Generate probability signals")
    parser.add_argument(
        "--simulate-closed-market",
        action="store_true",
        help="Force closed market behaviour for testing",
    )
    args = parser.parse_args()

    cfg = load_config()
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
    last_ts = None
    prev_models: list[str] = []
    if state:
        last_ts = state.get("last_timestamp")
        prev_models = state.get("model_versions", [])

    if args.simulate_closed_market or not is_market_open():
        logger.info("Market closed - running backtest and using historical data")
        backtest.run_rolling_backtest(cfg)

    model_type = cfg.get("model_type", "lgbm").lower()
    model_paths = cfg.get("ensemble_models", ["model.joblib"])
    model_versions = cfg.get("model_versions", [])
    env_version = os.getenv("MODEL_VERSION_ID")
    if env_version:
        model_versions.append(env_version)
    models, stored_features, meta_clf = load_models(
        model_paths, model_versions, return_meta=True
    )
    if not models and model_type != "autogluon":
        models = [joblib.load(Path(__file__).resolve().parent / "model.joblib")]

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
                    thr_map = _normalise_thresholds(payload.get("regime_thresholds"))
                    if thr_map:
                        setattr(online_model, "regime_thresholds", thr_map)
                        setattr(online_model, "regime_thresholds_", thr_map)
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
                    thr_map = _normalise_thresholds(
                        perf_section.get("regime_thresholds")
                    )
                    if thr_map:
                        setattr(online_model, "regime_thresholds", thr_map)
                        setattr(online_model, "regime_thresholds_", thr_map)
            logger.info("Loaded online model from %s", online_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load online model: %s", exc)

    regime_thresholds, default_threshold = _resolve_threshold_metadata(
        models, online_model, cfg
    )
    hist_path_pq = Path(__file__).resolve().parent / "data" / "history.parquet"
    if hist_path_pq.exists():
        df = load_history_parquet(hist_path_pq)
    else:
        cfg_root = Path(__file__).resolve().parent
        sym = cfg.get("symbol")
        df = load_history_config(sym, cfg, cfg_root)
        df.to_parquet(hist_path_pq, index=False)
    df = df[df.get("Symbol").isin([cfg.get("symbol")])]

    # Catch up on any missed ticks since last processed timestamp
    if last_ts is not None and "Timestamp" in df.columns:
        try:
            df = df[pd.to_datetime(df["Timestamp"]) > pd.to_datetime(last_ts)]
        except Exception:
            pass

    df = make_features(df)

    # optional macro indicators merged on timestamp
    macro_path = Path(__file__).resolve().parent / "data" / "macro.csv"
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

    if model_type == "autogluon":
        from autogluon.tabular import TabularPredictor

        ag_path = Path(__file__).resolve().parent / "models" / "autogluon"
        predictor = TabularPredictor.load(str(ag_path))

        def _predict(data: pd.DataFrame) -> np.ndarray:
            return predictor.predict_proba(data[features])[1].values

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
