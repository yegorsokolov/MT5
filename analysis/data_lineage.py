"""Utilities for tracking feature lineage.

This module records the relationship between raw input files, the
transformations applied to them and the resulting features.  Each record is
associated with a ``run_id`` so downstream consumers can query lineage
information for a particular model run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

# Path to the persistent lineage store.  The directory is created on demand.
STORE_PATH = Path(__file__).resolve().parents[1] / "lineage" / "lineage.parquet"
STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_store() -> pd.DataFrame:
    """Return the full lineage store as a DataFrame."""

    if STORE_PATH.exists():
        return pd.read_parquet(STORE_PATH)
    return pd.DataFrame(
        columns=["run_id", "raw_file", "transformation", "output_feature"]
    )


def _save_store(df: pd.DataFrame) -> None:
    """Persist ``df`` to the lineage store."""

    df.to_parquet(STORE_PATH, index=False)


def log_lineage(
    run_id: str, raw_file: str, transformation: str, output_feature: str
) -> None:
    """Append a lineage record for ``run_id``.

    Parameters
    ----------
    run_id:
        Identifier for the current model run.
    raw_file:
        Path or identifier of the raw data file used.
    transformation:
        Name of the transformation function applied.
    output_feature:
        Name of the resulting feature/column.
    """

    df = _load_store()
    record = {
        "run_id": run_id,
        "raw_file": raw_file,
        "transformation": transformation,
        "output_feature": output_feature,
    }
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    _save_store(df)


def get_lineage(run_id: str) -> pd.DataFrame:
    """Return lineage records for ``run_id``."""

    df = _load_store()
    return df[df["run_id"] == run_id].copy()


def search(**criteria: str) -> pd.DataFrame:
    """Return lineage records matching ``criteria``.

    Examples
    --------
    ``search(output_feature="close")`` will return all lineage entries that
    produced a feature named ``"close"`` regardless of run id.
    """

    df = _load_store()
    for key, value in criteria.items():
        if key in df.columns:
            df = df[df[key] == value]
    return df.copy()


__all__ = ["log_lineage", "get_lineage", "search"]

