from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Minimum capability tier required to use this module
min_capability = "standard"

# Allow tests to override storage locations
_DATA_PATH = Path(
    os.getenv(
        "NEWS_SENTIMENT_FUSED_PATH",
        Path(__file__).resolve().parents[1] / "analysis" / "news_sentiment_fused.parquet",
    )
)
_MODEL_PATH = Path(
    os.getenv(
        "NEWS_SENTIMENT_FUSION_MODEL",
        Path(__file__).resolve().with_suffix(".pkl"),
    )
)

_model_cache: RandomForestRegressor | None = None
_df_cache: pd.DataFrame | None = None


def _prepare_features(events: pd.DataFrame) -> np.ndarray:
    """Extract numeric features from raw event data."""

    emb = events["embedding"].apply(
        lambda v: float(np.mean(v)) if isinstance(v, Iterable) else float(v)
    )
    feats = np.column_stack([emb.to_numpy(), events["surprise"].to_numpy()])
    return feats


def train(events: pd.DataFrame, target: Iterable[float]) -> RandomForestRegressor:
    """Train the sentiment fusion model and persist per-event scores."""

    global _model_cache, _df_cache
    X = _prepare_features(events)
    y = np.asarray(list(target), dtype=float)
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    out = pd.DataFrame(
        {
            "symbol": events["symbol"],
            "timestamp": pd.to_datetime(events["timestamp"]),
            "fused_sentiment": preds,
        }
    )
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_DATA_PATH, index=False)
    joblib.dump(model, _MODEL_PATH)
    _model_cache = model
    _df_cache = out
    return model


def _load_model() -> RandomForestRegressor:
    global _model_cache
    if _model_cache is None:
        if _MODEL_PATH.exists():
            _model_cache = joblib.load(_MODEL_PATH)
        else:  # pragma: no cover - model missing
            _model_cache = RandomForestRegressor(n_estimators=50, random_state=0)
    return _model_cache


def score(events: pd.DataFrame) -> pd.DataFrame:
    """Return fused sentiment scores for ``events``."""

    model = _load_model()
    X = _prepare_features(events)
    scores = model.predict(X)
    return pd.DataFrame(
        {
            "symbol": events["symbol"],
            "timestamp": pd.to_datetime(events["timestamp"]),
            "fused_sentiment": scores,
        }
    )


def load_scores() -> pd.DataFrame:
    """Load persisted fused sentiment scores."""

    global _df_cache
    if _df_cache is None:
        if _DATA_PATH.exists():
            _df_cache = pd.read_parquet(_DATA_PATH)
        else:
            _df_cache = pd.DataFrame(columns=["symbol", "timestamp", "fused_sentiment"])
    return _df_cache


def get_sentiment(symbol: str, timestamp: str | pd.Timestamp) -> float | None:
    """Return fused sentiment for ``symbol`` at ``timestamp`` if available."""

    df = load_scores()
    if df.empty:
        return None
    ts = pd.Timestamp(timestamp)
    mask = (df["symbol"] == symbol) & (pd.to_datetime(df["timestamp"]) == ts)
    row = df[mask]
    if not row.empty:
        return float(row["fused_sentiment"].iloc[0])
    return None


__all__ = ["train", "score", "load_scores", "get_sentiment"]
