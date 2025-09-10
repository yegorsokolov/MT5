from __future__ import annotations

import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage change feature used for demonstration."""

    df = df.copy()
    df["plugin_feature"] = df["close"].pct_change().fillna(0)
    return df


def register(register_feature):
    """Entry point hook used by MT5 to register this feature."""

    register_feature("plugin_feature", compute)
