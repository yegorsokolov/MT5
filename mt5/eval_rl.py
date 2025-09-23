"""Evaluate a trained RL agent by replaying actions on historical data."""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from sb3_contrib.qrdqn import QRDQN

from utils import load_config
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
)
from data.features import make_features
from mt5.train_rl import TradingEnv, DiscreteTradingEnv, artifact_dir
from rl.multi_objective import pareto_frontier
from mt5.log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for RL evaluation."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def compute_metrics(returns: pd.Series) -> dict:
    """Compute Sharpe, drawdown and related metrics."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    std = returns.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        sharpe = 0.0
    else:
        sharpe = np.sqrt(252) * returns.mean() / std
    return {
        "sharpe": sharpe,
        "max_drawdown": drawdown.min() * 100,
        "total_return": cumulative.iloc[-1] - 1,
        "win_rate": (returns > 0).mean() * 100,
    }


def load_dataset(cfg: dict, root: Path) -> pd.DataFrame:
    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs: List[pd.DataFrame] = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root)
        df_sym["Symbol"] = sym
        dfs.append(df_sym)
    return make_features(pd.concat(dfs, ignore_index=True))


def feature_list(df: pd.DataFrame) -> List[str]:
    feats = [
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
    feats += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        feats.extend(["volume_ratio", "volume_imbalance"])
    return feats


@log_exceptions
def main() -> None:
    init_logging()
    cfg = load_config()
    root = artifact_dir(cfg)
    models_dir = root / "models"
    reports_dir = root / "reports"
    df = load_dataset(cfg, root)
    features = feature_list(df)

    algo = cfg.get("rl_algorithm", "PPO").upper()
    if algo == "PPO":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            objectives=cfg.get("rl_objectives", ["return"]),
            objective_weights=cfg.get("rl_objective_weights"),
        )
        model_cls = PPO
    elif algo == "SAC":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            objectives=cfg.get("rl_objectives", ["return"]),
            objective_weights=cfg.get("rl_objective_weights"),
        )
        model_cls = SAC
    elif algo == "QRDQN":
        env = DiscreteTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
            objectives=cfg.get("rl_objectives", ["return"]),
            objective_weights=cfg.get("rl_objective_weights"),
        )
        model_cls = QRDQN
    else:
        raise ValueError(f"Unknown rl_algorithm {algo}")

    model_path = models_dir / "model_rl.zip"
    if not model_path.exists():
        raise FileNotFoundError("model_rl.zip not found, please train the agent first")
    model = model_cls.load(model_path, env=env)

    obs = env.reset()
    done = False
    equities = [env.equity]
    returns: List[float] = []
    objective_sums = {name: 0.0 for name in getattr(env, "objectives", [])}
    reward_vectors: List[List[float]] = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev_eq = env.equity
        obs, _reward, done, info = env.step(action)
        returns.append(env.equity / prev_eq - 1)
        equities.append(env.equity)
        obj = info.get("objectives")
        if obj:
            for k, v in obj.items():
                objective_sums[k] += float(v)
            reward_vectors.append([obj.get(k, 0.0) for k in env.objectives])

    curve = pd.DataFrame({"step": range(len(equities)), "equity": equities})
    equity_curve_path = reports_dir / "equity_curve_rl.csv"
    curve.to_csv(equity_curve_path, index=False)
    metrics = compute_metrics(pd.Series(returns))

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        if k in {"max_drawdown", "win_rate"}:
            logger.info("%s: %.2f%%", k, v)
        else:
            logger.info("%s: %.4f", k, v)
    logger.info("Equity curve saved to %s", equity_curve_path)
    if objective_sums:
        logger.info("Objective trade-offs:")
        for k, v in objective_sums.items():
            logger.info("%s_total: %.4f", k, v)
        if reward_vectors:
            frontier = pareto_frontier(reward_vectors)
            logger.info("Pareto frontier: %s", frontier.tolist())
    if "market_regime" in env.df.columns:
        regime_series = env.df.loc[1:, "market_regime"].to_numpy()
        returns_arr = np.asarray(returns)
        for regime in np.unique(regime_series):
            mask = regime_series == regime
            if mask.any():
                r_metrics = compute_metrics(pd.Series(returns_arr[mask]))
                logger.info("Regime %s metrics: %s", regime, r_metrics)


if __name__ == "__main__":
    main()
