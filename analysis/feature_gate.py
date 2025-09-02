from __future__ import annotations

"""Feature gating based on mutual information importance.

This module computes simple mutual information scores for features segmented by
market regime and persists the selected feature set to ensure consistent gating
across runs.  At inference time the persisted list is loaded and low-importance
or heavy features are dropped depending on hardware capability tiers.
"""

from pathlib import Path
import json
import logging
from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)

# Heavy optional features that are expensive to compute or load.  These are
# always removed on constrained tiers.
HEAVY_FEATURES = {
    "implied_vol",
    "active_addresses",
    "esg_score",
    "shipping_metric",
    "retail_sales",
    "temperature",
    "revenue",
    "net_income",
    "pe_ratio",
    "dividend_yield",
    "gdp",
    "cpi",
    "interest_rate",
    "heavy_feat",  # used in tests
}

# Essential columns that should never be removed
ESSENTIAL = {
    "Timestamp",
    "Symbol",
    "Bid",
    "Ask",
    "return",
    "market_regime",
}

# Maximum number of features to keep for each capability tier
TIER_LIMITS = {"lite": 20, "standard": 50}

# Location for persisted feature selections
STORE_DIR = Path("models") / "feature_gates"


def _gate_path(regime: int, store_dir: str | Path | None = None) -> Path:
    store = Path(store_dir) if store_dir is not None else STORE_DIR
    store.mkdir(parents=True, exist_ok=True)
    return store / f"regime_{regime}.json"


def compute_importance(df: pd.DataFrame, regime: int) -> pd.Series:
    """Return mutual information scores for numeric features.

    Parameters
    ----------
    df:
        Feature dataframe containing ``return`` and ``market_regime`` columns.
    regime:
        Regime identifier used to filter ``df``.  When the column is missing or
        empty the entire dataframe is used.
    """

    if "market_regime" in df.columns:
        df = df[df["market_regime"] == regime]
    if df.empty or "return" not in df.columns:
        return pd.Series(dtype=float)

    candidates = [
        c
        for c in df.select_dtypes("number").columns
        if c not in ESSENTIAL and c != "return"
    ]
    if not candidates:
        return pd.Series(dtype=float)

    y = (df["return"].shift(-1) > 0).astype(int)
    X = df[candidates].fillna(0)
    y = y.iloc[: len(X)]
    if y.nunique() < 2:
        # Mutual information requires at least two classes
        return pd.Series(0.0, index=candidates)

    scores = mutual_info_classif(X, y, random_state=0)
    return pd.Series(scores, index=candidates).sort_values(ascending=False)


def select(
    df: pd.DataFrame,
    capability_tier: str,
    regime: int,
    store_dir: str | Path | None = None,
    persist: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop low-importance or heavy features from ``df``.

    The feature list is persisted under ``models/feature_gates`` (or ``store_dir``)
    keyed by ``regime`` so subsequent calls yield consistent feature sets.

    Parameters
    ----------
    df:
        Input feature dataframe.
    capability_tier:
        Hardware capability tier such as ``"lite"`` or ``"standard"``.
    regime:
        Regime identifier used for importance computation and persistence.
    store_dir:
        Optional custom directory for storing feature lists (primarily for
        tests).
    persist:
        If ``False`` the dataframe is filtered using existing persisted
        selections but no new importance computation is performed.  This is used
        by :mod:`data.features` to avoid expensive recalculations at runtime.
    """

    path = _gate_path(regime, store_dir)
    selected: List[str]
    if path.exists():
        selected = json.loads(path.read_text())
    elif persist:
        importance = compute_importance(df, regime)
        if not importance.empty:
            threshold = importance.mean()
            importance = importance[importance >= threshold]
            if importance.empty:
                importance = compute_importance(df, regime).head(1)
        limit = TIER_LIMITS.get(capability_tier)
        if limit is not None:
            importance = importance.head(limit)
        selected = importance.index.tolist()
        if capability_tier == "lite":
            selected = [f for f in selected if f not in HEAVY_FEATURES]
        path.write_text(json.dumps(selected))
    else:
        selected = []

    numeric_cols = [c for c in df.select_dtypes("number").columns if c not in ESSENTIAL]
    drop: List[str] = []
    if selected:
        drop = [c for c in numeric_cols if c not in selected]
    elif capability_tier == "lite":
        drop = [c for c in numeric_cols if c in HEAVY_FEATURES]

    result = df.drop(columns=drop, errors="ignore")
    return result, selected


__all__ = ["compute_importance", "select"]
