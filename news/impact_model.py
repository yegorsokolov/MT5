from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import numpy as np

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

_model_cache: GradientBoostingRegressor | None = None
_df_cache: pd.DataFrame | None = None
_residual_std: float = 0.0


def train(events: pd.DataFrame, target: pd.Series) -> GradientBoostingRegressor:
    """Train the impact model and persist per-event scores."""
    global _model_cache, _df_cache, _residual_std
    features = events[["surprise", "sentiment", "historical_response"]]
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
    _model_cache = model
    _df_cache = out
    return model


def _load_model() -> GradientBoostingRegressor:
    global _model_cache, _residual_std
    if _model_cache is None:
        if _MODEL_PATH.exists():
            obj = joblib.load(_MODEL_PATH)
            if isinstance(obj, dict):
                _model_cache = obj.get("model")
                _residual_std = obj.get("residual_std", 0.0)
            else:
                _model_cache = obj
                _residual_std = 0.0
        else:  # pragma: no cover - only when model missing
            _model_cache = GradientBoostingRegressor()
            _residual_std = 0.0
    return _model_cache


def score(events: pd.DataFrame) -> pd.DataFrame:
    """Return impact predictions for ``events``."""
    model = _load_model()
    feats = events[["surprise", "sentiment", "historical_response"]]
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
