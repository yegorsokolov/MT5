from __future__ import annotations

import warnings
from typing import Iterable, Sequence, Optional

import pandas as pd


def require_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    on_missing: str = "raise",
    fill_value: Optional[object] = None,
) -> pd.DataFrame:
    """Ensure ``df`` contains all ``columns``.

    Parameters
    ----------
    df:
        DataFrame to validate.
    columns:
        Required column names.
    on_missing:
        Behaviour when columns are missing. ``"raise"`` (default) raises
        :class:`ValueError`, ``"warn"`` adds the missing columns filled with
        ``fill_value`` and emits a warning, ``"ignore"`` silently continues.
    fill_value:
        Value used when ``on_missing='warn'`` to create missing columns.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        if on_missing == "warn":
            warnings.warn(msg)
            for col in missing:
                df[col] = fill_value
        elif on_missing != "ignore":
            raise ValueError(msg)
    return df


def assert_no_nan(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    *,
    on_error: str = "raise",
    fill_value: object = 0,
) -> pd.DataFrame:
    """Validate that specified ``columns`` contain no NaN values.

    Parameters
    ----------
    df:
        DataFrame to validate.
    columns:
        Columns to inspect. If ``None`` all columns are checked.
    on_error:
        Behaviour when NaNs are found. ``"raise"`` (default) raises
        :class:`ValueError`, ``"warn"`` replaces NaNs with ``fill_value`` and
        emits a warning, ``"ignore"`` silently continues.
    fill_value:
        Replacement value used when ``on_error='warn'``.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    nan_cols = [c for c in cols if df[c].isna().any()]
    if nan_cols:
        msg = f"NaN values found in columns: {nan_cols}"
        if on_error == "warn":
            warnings.warn(msg)
            df[nan_cols] = df[nan_cols].fillna(fill_value)
        elif on_error != "ignore":
            raise ValueError(msg)
    return df


__all__ = ["require_columns", "assert_no_nan"]
