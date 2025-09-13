from __future__ import annotations

"""Load evolved symbolic feature formulas and compute their values."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

FORMULA_PATH = Path(__file__).resolve().parents[1] / "analysis" / "evolved_symbols.yaml"


def _load_formulas(path: Path = FORMULA_PATH) -> List[Dict[str, Any]]:
    """Return stored symbolic feature formulas from ``path``."""

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
    """Append evolved symbolic feature columns defined in ``path``."""

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
            out[name] = eval(expr, ns)
        except Exception:
            continue
    return out


__all__ = ["compute"]
