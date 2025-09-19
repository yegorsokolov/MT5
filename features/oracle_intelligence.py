"""Features derived from prediction market oracles."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from oracles.oracle_scalper import OracleScalper

try:  # pragma: no cover - decorator optional in some environments
    from . import validate_module
except Exception:  # pragma: no cover - fallback without validation
    def validate_module(func):
        return func

try:  # pragma: no cover - optional validators
    from .validators import require_columns
except Exception:  # pragma: no cover - fallback when validators unavailable
    def require_columns(df, cols, **_):  # type: ignore[unused-arg]
        return df


logger = logging.getLogger(__name__)


def _alias_mapping(df: pd.DataFrame) -> Mapping[str, Iterable[str]] | None:
    aliases = df.attrs.get("oracle_aliases")
    if isinstance(aliases, Mapping):
        return aliases
    return None


@validate_module
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Augment ``df`` with oracle-derived probability features."""

    df = df.copy()
    require_columns(df, ["Timestamp"])

    scalper = OracleScalper()
    alias_map = _alias_mapping(df)

    try:
        augmented = scalper.augment_dataframe(df, aliases=alias_map)
    except Exception:  # pragma: no cover - network / parsing failures
        logger.warning("Oracle scalper failed, returning NaN features", exc_info=True)
        for column in scalper.feature_columns:
            if column not in df.columns:
                df[column] = np.nan
        return df

    for column in scalper.feature_columns:
        if column not in augmented.columns:
            augmented[column] = np.nan

    return augmented


__all__ = ["compute"]

