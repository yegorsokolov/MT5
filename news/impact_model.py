from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import numpy as np
from models import LazyModel
from . import vector_store

# Paths for persisted artefacts.  ``NEWS_IMPACT_PATH`` allows tests or
# callers to override the default location of the impact scores file.
_DATA_PATH = Path(
    os.getenv(
        "NEWS_IMPACT_PATH",
        Path(__file__).resolve().parents[1] / "analysis" / "news_impact.parquet",
    )
)
_MODEL_PATH = Path(
    os.getenv(
        "NEWS_IMPACT_MODEL",
        Path(__file__).resolve().with_suffix(".pkl"),
    )
)

_df_cache: pd.DataFrame | None = None
_residual_std: float = 0.0


def _model_loader(path: Path | str) -> GradientBoostingRegressor:
    """Load model and residual information from ``_MODEL_PATH``."""

    global _residual_std
    p = Path(_MODEL_PATH)
    if p.exists():
        obj = joblib.load(p)
        if isinstance(obj, dict):
            _residual_std = obj.get("residual_std", 0.0)
            return obj.get("model")  # type: ignore[return-value]
        _residual_std = 0.0
        return obj
    _residual_std = 0.0
    return GradientBoostingRegressor()


_MODEL = LazyModel(loader=_model_loader)


def _similarity_scores(texts: pd.Series, k: int = 5) -> pd.Series:
    """Return average similarity of each text to the existing vector store."""

    def score(t: str) -> float:
        sims = [s for _, s in vector_store.similar_events(t, k)] if t else []
        return float(np.mean(sims)) if sims else 0.0

    return texts.fillna("").apply(score)


def train(events: pd.DataFrame, target: pd.Series) -> GradientBoostingRegressor:
    """Train the impact model and persist per-event scores."""

    global _df_cache, _residual_std
    text_col = (
        events["event"]
        if "event" in events.columns
        else events.get("text", pd.Series([""] * len(events)))
    )
    features = pd.DataFrame(
        {
            "surprise": events["surprise"],
            "sentiment": events["sentiment"],
            "historical_response": events["historical_response"],
            "similarity": _similarity_scores(text_col),
        }
    )
    model = GradientBoostingRegressor()
    model.fit(features, target)
    preds = model.predict(features)
    _residual_std = float(np.std(target - preds)) if len(target) > 1 else 0.0
    out = pd.DataFrame(
        {
            "symbol": events["symbol"],
            "timestamp": pd.to_datetime(events["timestamp"]),
            "impact": preds,
            "uncertainty": _residual_std,
        }
    )
    out.to_parquet(_DATA_PATH, index=False)
    joblib.dump({"model": model, "residual_std": _residual_std}, _MODEL_PATH)
    _MODEL.set(model)
    _df_cache = out
    return model


def score(events: pd.DataFrame) -> pd.DataFrame:
    """Return impact predictions for ``events``."""

    model = _MODEL.load()
    text_col = (
        events["event"]
        if "event" in events.columns
        else events.get("text", pd.Series([""] * len(events)))
    )
    feats = pd.DataFrame(
        {
            "surprise": events["surprise"],
            "sentiment": events["sentiment"],
            "historical_response": events["historical_response"],
            "similarity": _similarity_scores(text_col),
        }
    )
    impact = model.predict(feats)
    return pd.DataFrame(
        {
            "symbol": events["symbol"],
            "timestamp": pd.to_datetime(events["timestamp"]),
            "impact": impact,
            "uncertainty": _residual_std,
        }
    )


def _load_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if _DATA_PATH.exists():
            _df_cache = pd.read_parquet(_DATA_PATH)
        else:
            _df_cache = pd.DataFrame(
                columns=["symbol", "timestamp", "impact", "uncertainty"]
            )
    return _df_cache


def get_impact(symbol: str, timestamp: str | pd.Timestamp) -> Tuple[float | None, float]:
    """Return impact and uncertainty for ``symbol`` at ``timestamp``."""
    df = _load_df()
    if df.empty:
        return None, _residual_std
    ts = pd.Timestamp(timestamp)
    mask = (df["symbol"] == symbol) & (pd.to_datetime(df["timestamp"]) == ts)
    row = df[mask]
    if not row.empty:
        return (
            float(row["impact"].iloc[0]),
            float(row.get("uncertainty", _residual_std).iloc[0]),
        )
    return None, _residual_std


__all__ = ["train", "score", "get_impact"]
