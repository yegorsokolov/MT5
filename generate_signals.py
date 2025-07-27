"""Generate per-tick probability signals for the EA."""

from pathlib import Path
import joblib
import pandas as pd

import numpy as np

from utils import load_config
from dataset import load_history_parquet, make_features, load_history_config
from train_rl import TradingEnv, DiscreteTradingEnv, RLLibTradingEnv
from stable_baselines3 import PPO, SAC
from sb3_contrib.qrdqn import QRDQN
from sklearn.linear_model import LogisticRegression
import asyncio
from signal_queue import (
    get_async_publisher,
    publish_dataframe_async,
)

from log_utils import setup_logging, log_exceptions

logger = setup_logging()


def load_models(paths):
    """Load multiple joblib models from relative paths."""
    models = []
    for p in paths:
        mp = Path(__file__).resolve().parent / p
        if mp.exists():
            models.append(joblib.load(mp))
    return models


def bayesian_average(prob_arrays):
    """Combine probabilities using a simple Bayesian model averaging."""
    logits = [np.log(p / (1 - p + 1e-12)) for p in prob_arrays]
    avg_logit = np.mean(logits, axis=0)
    return 1 / (1 + np.exp(-avg_logit))


def rl_signals(df, features, cfg):
    """Return probability-like signals from a trained RL agent."""
    model_path = Path(__file__).resolve().parent / "model_rl.zip"
    model_rllib = Path(__file__).resolve().parent / "model_rllib"
    algo = cfg.get("rl_algorithm", "PPO").upper()
    if algo == "RLLIB":
        if not model_rllib.exists():
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
        )
        model = PPO.load(model_path, env=env)
    elif algo == "SAC":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
        )
        model = SAC.load(model_path, env=env)
    elif algo == "QRDQN":
        env = DiscreteTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
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
        )
        model = QRDQN.load(model_path, env=env)

    if algo == "RLLIB":
        obs, _ = env.reset()
    else:
        obs = env.reset()
    done = False
    actions = []
    while not done:
        if algo == "RLLIB":
            action = model.compute_single_action(obs)
        else:
            action, _ = model.predict(obs, deterministic=True)
        a = float(action[0]) if not np.isscalar(action) else float(action)
        actions.append(a)
        if algo == "RLLIB":
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            obs, _, done, _ = env.step(action)

    probs = (np.array(actions) > 0).astype(float)
    if len(probs) < len(df):
        probs = np.pad(probs, (0, len(df) - len(probs)), "edge")
    if algo == "RLLIB":
        ray.shutdown()
    return probs


@log_exceptions
def main():
    cfg = load_config()

    model_paths = cfg.get("ensemble_models", ["model.joblib"])
    models = load_models(model_paths)
    if not models:
        models = [joblib.load(Path(__file__).resolve().parent / "model.joblib")]
    hist_path_pq = Path(__file__).resolve().parent / "data" / "history.parquet"
    if hist_path_pq.exists():
        df = load_history_parquet(hist_path_pq)
    else:
        cfg_root = Path(__file__).resolve().parent
        sym = cfg.get("symbol")
        df = load_history_config(sym, cfg, cfg_root)
        df.to_parquet(hist_path_pq, index=False)
    df = df[df.get("Symbol").isin([cfg.get("symbol")])]
    df = make_features(df)

    # optional macro indicators merged on timestamp
    macro_path = Path(__file__).resolve().parent / "data" / "macro.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path)
        macro["Timestamp"] = pd.to_datetime(macro["Timestamp"])
        df = df.merge(macro, on="Timestamp", how="left")
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

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

    prob_list = [m.predict_proba(df[features])[:, 1] for m in models]
    if len(prob_list) == 1:
        probs = prob_list[0]
    else:
        method = cfg.get("ensemble_method", "average")
        if method == "bayesian":
            probs = bayesian_average(prob_list)
        else:
            probs = np.mean(prob_list, axis=0)

    if cfg.get("blend_with_rl", False):
        rl_probs = rl_signals(df, features, cfg)
        X = np.column_stack([probs, rl_probs])
        y = (df["return"].shift(-1) > 0).astype(int)
        train_n = int(len(X) * 0.7)
        lr = LogisticRegression()
        lr.fit(X[:train_n], y[:train_n])
        probs = lr.predict_proba(X)[:, 1]

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

    out = pd.DataFrame({"Timestamp": df["Timestamp"], "prob": combined})
    pub = get_async_publisher()
    asyncio.run(publish_dataframe_async(pub, out))
    print("Signals published via ZeroMQ (async)")


if __name__ == "__main__":
    main()
