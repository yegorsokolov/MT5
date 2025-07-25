"""Evaluate a trained RL agent by replaying actions on historical data."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib.qrdqn import QRDQN

from utils import load_config
from dataset import (
    load_history_parquet,
    save_history_parquet,
    make_features,
    load_history_config,
)
from train_rl import TradingEnv, DiscreteTradingEnv
from log_utils import setup_logging, log_exceptions

logger = setup_logging()


def compute_metrics(returns: pd.Series) -> dict:
    """Compute Sharpe, drawdown and related metrics."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
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
    cfg = load_config()
    root = Path(__file__).resolve().parent
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
        )
        model_cls = PPO
    elif algo == "QRDQN":
        env = DiscreteTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
        )
        model_cls = QRDQN
    else:
        raise ValueError(f"Unknown rl_algorithm {algo}")

    model_path = root / "model_rl.zip"
    if not model_path.exists():
        raise FileNotFoundError("model_rl.zip not found, please train the agent first")
    model = model_cls.load(model_path, env=env)

    obs = env.reset()
    done = False
    equities = [env.equity]
    returns: List[float] = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev_eq = env.equity
        obs, _reward, done, _info = env.step(action)
        returns.append(env.equity / prev_eq - 1)
        equities.append(env.equity)

    curve = pd.DataFrame({"step": range(len(equities)), "equity": equities})
    curve.to_csv(root / "equity_curve_rl.csv", index=False)
    metrics = compute_metrics(pd.Series(returns))

    print("Evaluation metrics:")
    for k, v in metrics.items():
        if k in {"max_drawdown", "win_rate"}:
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.4f}")
    print("Equity curve saved to", root / "equity_curve_rl.csv")


if __name__ == "__main__":
    main()
