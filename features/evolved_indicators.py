from __future__ import annotations

"""Load evolved indicator formulas and compute their values."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Default location where evolved indicator formulas are stored.  The files are
# versioned using the ``evolved_indicators_v*.json`` pattern within the feature
# store.  ``FORMULA_PATH`` points at the first version so that the module works
# out-of-the-box while still allowing callers to override the path.
FORMULA_PATH = (
    Path(__file__).resolve().parents[1] / "feature_store" / "evolved_indicators_v1.json"
)


def _load_formulas(path: Path = FORMULA_PATH) -> List[Dict[str, Any]]:
    """Return stored indicator formulas from ``path``.

    The files are JSON encoded and typically live in the ``feature_store``
    directory.  The function is resilient to missing or malformed files,
    returning an empty list in those cases.
    """

    try:
        text = path.read_text()
    except Exception:
        return []
    if not text.strip():
        return []
    try:
        return json.loads(text)
    except Exception:
        return []


def compute(df: pd.DataFrame, path: Path = FORMULA_PATH) -> pd.DataFrame:
    """Append evolved indicator columns defined in ``path``.

    Each entry in the file must provide ``name`` and ``formula`` fields.  The
    formula is evaluated with ``df`` in scope and may reference ``numpy`` as
    ``np`` and ``pandas`` as ``pd``.
    """

    formulas = _load_formulas(path)
    if not formulas:
        return df
    out = df.copy()
    ns = {"np": np, "pd": pd, "df": out}
    for item in formulas:
        name = item.get("name")
        expr = item.get("formula")
        if not name or not expr:
            continue
        try:
            result = eval(expr, ns)
        except Exception:
            continue
        if isinstance(result, pd.Series):
            series = result.reindex(out.index)
        elif isinstance(result, np.ndarray):
            arr = np.asarray(result)
            if arr.ndim == 0:
                out[name] = arr.item()
                ns[name] = out[name]
                continue
            if len(arr) != len(out):
                continue
            series = pd.Series(arr, index=out.index)
        elif isinstance(result, (list, tuple)):
            if len(result) != len(out):
                continue
            series = pd.Series(result, index=out.index)
        else:
            out[name] = result
            ns[name] = out[name]
            continue
        out[name] = series
        ns[name] = out[name]
    return out


__all__ = ["compute"]
