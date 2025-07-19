"""Meta-learning style training with symbol-specific adapters."""

from pathlib import Path
from typing import List
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from utils import load_config
from dataset import (
    load_history,
    load_history_from_urls,
    make_features,
)


def load_symbol_data(sym: str, cfg: dict, root: Path) -> pd.DataFrame:
    """Load history for a single symbol, downloading if needed."""
    path = root / "data" / f"{sym}_history.csv"
    if path.exists():
        df = load_history(path)
    else:
        urls = cfg.get("data_urls", {}).get(sym)
        if not urls:
            raise FileNotFoundError(f"No history found for {sym} and no URL configured")
        df = load_history_from_urls(urls)
        df.to_csv(path, index=False)
    df["Symbol"] = sym
    return df


def train_base_model(df: pd.DataFrame, features: List[str]):
    """Train a global base model on all symbols."""
    X = df[features]
    y = (df["return"].shift(-1) > 0).astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=200, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


def train_symbol_adapters(
    df: pd.DataFrame, features: List[str], base_model, out_dir: Path, symbols: List[str]
):
    """Train lightweight adapters per symbol using base model probabilities."""
    out_dir.mkdir(exist_ok=True)
    for sym in symbols:
        df_sym = df[df["Symbol"] == sym].copy()
        X_sym = df_sym[features]
        y_sym = (df_sym["return"].shift(-1) > 0).astype(int)
        base_probs = base_model.predict_proba(X_sym)[:, 1]
        X_meta = X_sym.copy()
        X_meta["base_prob"] = base_probs
        adapter = LogisticRegression(max_iter=200)
        adapter.fit(X_meta, y_sym)
        joblib.dump(adapter, out_dir / f"model_{sym}.joblib")


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    symbols = cfg.get("symbols") or [cfg.get("symbol")]

    # load and combine histories
    dfs = [load_symbol_data(s, cfg, root) for s in symbols]
    df = make_features(pd.concat(dfs, ignore_index=True))
    df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    features.append("SymbolCode")

    base_model = train_base_model(df, features)
    joblib.dump(base_model, root / "models" / "base_model.joblib")

    train_symbol_adapters(df, features, base_model, root / "models", symbols)
    print("Models saved to", root / "models")


if __name__ == "__main__":
    main()
