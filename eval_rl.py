"""Evaluate a trained RL agent by replaying actions on historical data."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from utils import load_config
from dataset import (
    load_history,
    load_history_from_urls,
    load_history_parquet,
    save_history_parquet,
    make_features,
)
from train_rl import TradingEnv
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
        csv_path = root / "data" / f"{sym}_history.csv"
        pq_path = root / "data" / f"{sym}_history.parquet"
        if pq_path.exists():
            df_sym = load_history_parquet(pq_path)
        elif csv_path.exists():
            df_sym = load_history(csv_path)
        else:
            urls = cfg.get("data_urls", {}).get(sym)
            if not urls:
                raise FileNotFoundError(
                    f"No history found for {sym} and no URL configured"
                )
            df_sym = load_history_from_urls(urls)
            save_history_parquet(df_sym, pq_path)
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

    env = TradingEnv(
        df,
        features,
        max_position=cfg.get("rl_max_position", 1.0),
        transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
        risk_penalty=cfg.get("rl_risk_penalty", 0.1),
        var_window=cfg.get("rl_var_window", 30),
    )
    model_path = root / "model_rl.zip"
    if not model_path.exists():
        raise FileNotFoundError("model_rl.zip not found, please train the agent first")
    model = PPO.load(model_path, env=env)

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
