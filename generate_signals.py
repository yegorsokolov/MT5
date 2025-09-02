"""Generate per-tick probability signals for the EA."""

from log_utils import setup_logging, log_exceptions, log_predictions

from pathlib import Path
import os
import logging
import joblib
import pandas as pd
from state_manager import load_runtime_state, save_runtime_state

import numpy as np

from utils import load_config
from prediction_cache import PredictionCache
from typing import Any
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
from signal_queue import (
    get_async_publisher,
    publish_dataframe_async,
    get_signal_backend,
    get_publisher,
    request_history,
)
from models.ensemble import EnsembleModel
from models import model_store

setup_logging()
logger = logging.getLogger(__name__)


def fetch_history(symbol: str, timeframe: str, start: str, end: str, path: str = "history_request.csv") -> pd.DataFrame:
    """Helper to request historical rates from the EA."""
    with get_publisher() as sock:
        return request_history(sock, symbol, timeframe, start, end, path)


def load_models(paths, versions=None):
    """Load multiple models from paths or version identifiers."""
    models = []
    feature_list = None
    versions = versions or []
    for vid in versions:
        try:
            m, meta = model_store.load_model(vid)
            models.append(m)
            if feature_list is None:
                feature_list = meta.get("features") or meta.get("training_config", {}).get(
                    "features"
                )
        except FileNotFoundError:
            logger.warning("Model version %s not found", vid)
    for p in paths:
        mp = Path(__file__).resolve().parent / p
        if mp.exists():
            models.append(joblib.load(mp))
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
    model = MultiHeadTransformer(len(feat), num_symbols=num_symbols, num_regimes=num_regimes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    tier = monitor.capabilities.capability_tier()
    TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}
    sym_code = int(df.get("SymbolCode", pd.Series([0])).iloc[0]) if "SymbolCode" in df.columns else 0
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
        Path(__file__).resolve().parent / "models" / "recurrent_rl" / "recurrent_model.zip"
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
    cache = PredictionCache(
        cfg.get("pred_cache_size", 256), cfg.get("pred_cache_policy", "lru")
    )

    # Reload previous runtime state if available
    state = load_runtime_state()
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
    models, stored_features = load_models(model_paths, model_versions)
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
    online_path = Path(__file__).resolve().parent / "models" / "online.joblib"
    if cfg.get("use_online_model", False) and online_path.exists():
        try:
            online_model, _ = joblib.load(online_path)
            logger.info("Loaded online model from %s", online_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load online model: %s", exc)
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
            def _gbm_predict(data: pd.DataFrame) -> np.ndarray:
                return np.mean(
                    [m.predict_proba(data[features])[:, 1] for m in models], axis=0
                )

            base_models["lightgbm"] = _gbm_predict

        if cfg.get("use_meta_model", False):
            base_models["transformer"] = lambda d: meta_transformer_signals(
                d, features, cfg
            )

        if online_model is not None:
            def _online_predict(data: pd.DataFrame) -> np.ndarray:
                return np.array([
                    online_model.predict_proba_one(row).get(1, 0.0)
                    for row in data[features].to_dict("records")
                ])

            base_models["online"] = _online_predict

        if cfg.get("blend_with_rl", False):
            base_models["rl"] = lambda d: rl_signals(d, features, cfg)

        ensemble = EnsembleModel(base_models) if base_models else None

        def _predict(data: pd.DataFrame) -> np.ndarray:
            if ensemble is None:
                return np.zeros(len(data))
            return ensemble.predict(data)["ensemble"]

    hashes = pd.util.hash_pandas_object(df[features], index=False).values
    probs = np.zeros(len(df))
    pred_dict = {"ensemble": probs}
    miss_idx: list[int] = []
    for i, h in enumerate(hashes):
        val = cache.get(int(h))
        if val is not None:
            probs[i] = val
        else:
            miss_idx.append(i)
    if miss_idx:
        sub_df = df.iloc[miss_idx]
        new_probs = _predict(sub_df)
        for j, idx in enumerate(miss_idx):
            prob = float(new_probs[j])
            probs[idx] = prob
            cache.set(int(hashes[idx]), prob)

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

    combined = np.where(
        ma_ok & rsi_ok & boll_ok & vol_ok & macro_ok & news_ok & sent_ok & mom_ok,
        probs,
        0.0,
    )

    out = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Symbol": cfg.get("symbol"),
        "prob": combined,
    })
    log_df = df[["Timestamp"] + features].copy()
    log_df["Symbol"] = cfg.get("symbol")
    for name, arr in pred_dict.items():
        log_df[f"prob_{name}"] = arr
    log_df["prob"] = combined
    log_predictions(log_df)
    fmt = os.getenv("SIGNAL_FORMAT", "protobuf")
    backend = cfg.get("signal_backend", "zmq").lower()

    if backend == "zmq":
        async def _publish():
            async with get_async_publisher() as pub:
                await publish_dataframe_async(pub, out, fmt=fmt)

        asyncio.run(_publish())
        logger.info("Signals published via ZeroMQ (%s)", fmt)
    else:
        queue = get_signal_backend(cfg)
        if queue is None:
            raise ValueError(f"Unknown signal backend: {backend}")
        queue.publish_dataframe(out, fmt=fmt)
        queue.close()
        logger.info("Signals published via %s (%s)", backend, fmt)

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
        save_runtime_state(last_processed, open_positions, model_versions)
    except Exception:
        logger.exception("Failed to persist runtime state")


if __name__ == "__main__":
    main()
