"""Cointegration-based pair features.

This module computes Engle–Granger cointegration tests and rolling
spread z-scores for configured symbol pairs.  It expects a dataframe
containing at least ``Timestamp``, ``Symbol`` and a price column such as
``Close``, ``mid`` or ``Bid``.  For each configured pair ``(A, B)`` the
function adds the following columns:

``pair_z_A_B``
    Rolling z-score of the spread ``A - beta * B`` where ``beta`` is the
    hedge ratio estimated via ordinary least squares.
``hedge_A_B``
    Estimated hedge ratio ``beta`` used for the spread.
``coint_p_A_B``
    P-value of the Engle–Granger cointegration test between the two
    series.

Pairs are specified either via the ``pairs`` argument or through the
configuration returned by :func:`utils.load_config` under the key
``cointegration.pairs``.  Missing or invalid pairs are silently skipped.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

try:  # pragma: no cover - decorator optional in some tests
    from . import validate_module
except Exception:  # pragma: no cover - fallback when imported directly

    def validate_module(func):  # type: ignore
        return func


Pair = Tuple[str, str]


@validate_module
def compute(
    df: pd.DataFrame,
    pairs: Iterable[Pair] | None = None,
    window: int = 20,
) -> pd.DataFrame:
    """Append cointegration features for symbol ``pairs``.

    Parameters
    ----------
    df:
        Input dataframe containing price information.
    pairs:
        Iterable of symbol pairs.  If ``None``, pairs are loaded from the
        configuration via :func:`utils.load_config` under
        ``cointegration.pairs``.
    window:
        Rolling window used for the z-score of the spread.
    """

    if pairs is None:
        try:  # pragma: no cover - config optional for standalone tests
            from utils import load_config

            cfg = load_config()
            pair_cfg: List[Tuple[str, str]] = [
                tuple(p) for p in cfg.get("cointegration", {}).get("pairs", [])
            ]
            window = cfg.get("cointegration", {}).get("window", window)
        except Exception:
            pair_cfg = []
        pairs = pair_cfg
    else:
        pairs = list(pairs)

    if not pairs:
        return df

    df = df.copy().sort_values("Timestamp")
    price_col = "Close"
    if price_col not in df.columns:
        price_col = "mid" if "mid" in df.columns else "Bid"
    wide = df.pivot(index="Timestamp", columns="Symbol", values=price_col).sort_index()

    for s1, s2 in pairs:
        if s1 not in wide.columns or s2 not in wide.columns:
            continue
        y = wide[s1]
        x = wide[s2]
        aligned = pd.concat([y, x], axis=1).dropna()
        if aligned.empty:
            continue
        y_a = aligned[s1]
        x_a = aligned[s2]
        try:
            beta = float(np.polyfit(x_a.values, y_a.values, 1)[0])
        except Exception:
            beta = 1.0
        spread = y - beta * x
        z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
        try:
            _score, pvalue, _ = coint(y_a, x_a)
        except Exception:
            pvalue = np.nan

        zname = f"pair_z_{s1}_{s2}"
        df[zname] = z.reindex(df["Timestamp"]).values
        df[zname] = df[zname].fillna(0.0)
        df[f"hedge_{s1}_{s2}"] = beta
        df[f"coint_p_{s1}_{s2}"] = float(pvalue)

    return df


__all__ = ["compute"]
