"""Unified training pipeline combining transformer, meta-learning and RL."""

from __future__ import annotations

from pathlib import Path
from typing import List

import logging
import joblib
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO, SAC
from sb3_contrib.qrdqn import QRDQN

from utils import load_config
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
)
from data.features import (
    make_features,
    train_test_split,
    make_sequence_arrays,
)
from mt5.log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for the combined training pipeline."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)
from mt5.train_meta import load_symbol_data, train_base_model, train_meta_network
from mt5.train_rl import TradingEnv, DiscreteTradingEnv
from mt5.train_nn import TransformerModel


def load_dataset(cfg: dict, root: Path) -> pd.DataFrame:
    """Load history for all symbols and generate features."""
    root.joinpath("data").mkdir(exist_ok=True)
    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root, validate=cfg.get("validate", False))
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = make_features(pd.concat(dfs, ignore_index=True), validate=cfg.get("validate", False))
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    return df


def train_transformer(df: pd.DataFrame, cfg: dict, root: Path) -> None:
    """Train the transformer sequence model and save weights."""
    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "market_regime",
    ]
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

    seq_len = cfg.get("sequence_length", 50)
    train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))
    X_train, y_train = make_sequence_arrays(train_df, features, seq_len)
    X_test, y_test = make_sequence_arrays(test_df, features, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_symbols = int(df["Symbol"].nunique()) if "Symbol" in df.columns else None
    num_regimes = (
        int(df["market_regime"].nunique()) if "market_regime" in df.columns else None
    )
    model = TransformerModel(
        len(features),
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        num_symbols=num_symbols,
        num_regimes=num_regimes,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    for epoch in range(cfg.get("epochs", 5)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            pred_labels = (preds > 0.5).int()
            correct += (pred_labels == yb.int()).sum().item()
            total += yb.size(0)
    logger.info("Transformer accuracy: %s", correct / total)

    joblib.dump(model.state_dict(), root / "model_transformer.pt")
    logger.info("Transformer model saved to %s", root / "model_transformer.pt")


def run_meta_learning(df: pd.DataFrame, cfg: dict, root: Path) -> None:
    """Train base model and neural meta adapter."""
    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "market_regime",
    ]
    features += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    features.append("SymbolCode")

    base_model = train_base_model(df, features, cfg)
    joblib.dump(base_model, root / "models" / "base_model.joblib")
    train_meta_network(df, features, base_model, root / "models" / "meta_adapter.pt")
    logger.info(
        "Meta-learning model saved to %s", root / "models" / "meta_adapter.pt"
    )


def train_rl_agent(df: pd.DataFrame, cfg: dict, root: Path) -> None:
    """Train an RL agent using the trading environment."""
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
    features += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    seed = cfg.get("seed", 42)
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
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    elif algo == "SAC":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
        )
        model = SAC("MlpPolicy", env, verbose=0, seed=seed)
    elif algo == "QRDQN":
        env = DiscreteTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
        )
        model = QRDQN("MlpPolicy", env, verbose=0, seed=seed)
    else:
        raise ValueError(f"Unknown rl_algorithm {algo}")
    model.learn(total_timesteps=cfg.get("rl_steps", 5000))
    model.save(root / "model_rl")
    logger.info("RL model saved to %s", root / "model_rl.zip")


def evaluate_filters(df: pd.DataFrame, cfg: dict, root: Path) -> List[str]:
    """Analyse filter profitability and store the efficient ones."""
    results: dict[str, float] = {}
    future_ret = df["return"].shift(-1)

    if "ma_cross" in df.columns:
        mask = df["ma_cross"] == 1
        results["ma_cross"] = future_ret[mask].mean()

    if "rsi_14" in df.columns:
        mask = df["rsi_14"] > cfg.get("rsi_buy", 55)
        results["rsi_14"] = future_ret[mask].mean()

    if "boll_break" in df.columns:
        mask = df["boll_break"] == 1
        results["boll_break"] = future_ret[mask].mean()

    if "volume_spike" in df.columns:
        mask = df["volume_spike"] == 1
        results["volume_spike"] = future_ret[mask].mean()

    if "macro_indicator" in df.columns:
        mask = df["macro_indicator"] > cfg.get("macro_threshold", 0.0)
        results["macro_indicator"] = future_ret[mask].mean()

    if "nearest_news_minutes" in df.columns:
        window = cfg.get("avoid_news_minutes", 5)
        mask = df["nearest_news_minutes"] > window
        results["news_filter"] = future_ret[mask].mean()

    keep = [name for name, val in results.items() if pd.notna(val) and val > 0]
    (root / "active_filters.txt").write_text("\n".join(keep))
    logger.info("Filter performance: %s", results)
    logger.info("Keeping filters: %s", keep)
    return keep


@log_exceptions
def main() -> None:
    init_logging()
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root = Path(__file__).resolve().parent
    df = load_dataset(cfg, root)

    train_transformer(df, cfg, root)
    run_meta_learning(df, cfg, root)
    train_rl_agent(df, cfg, root)
    evaluate_filters(df, cfg, root)


if __name__ == "__main__":
    main()
