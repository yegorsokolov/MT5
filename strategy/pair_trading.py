from __future__ import annotations

"""Simple pair trading utilities.

This module detects cointegrated pairs using the Engle–Granger
cointegration test and generates z-score based trading signals with a
dynamic hedge ratio estimated via ordinary least squares.
It operates on tick history data as produced by :mod:`data.history`.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

Pair = Tuple[str, str, float]


def find_cointegrated_pairs(df: pd.DataFrame, significance: float = 0.05) -> List[Pair]:
    """Return all symbol pairs that are cointegrated."""
    if df.empty:
        return []
    prices = df.pivot(index="Timestamp", columns="Symbol", values="Bid").dropna()
    symbols = list(prices.columns)
    pairs: List[Pair] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1, s2 = symbols[i], symbols[j]
            try:
                _score, pvalue, _ = coint(prices[s1], prices[s2])
            except Exception:
                continue
            if pvalue < significance:
                pairs.append((s1, s2, float(pvalue)))
    return pairs


def _hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Return OLS hedge ratio between ``y`` and ``x``."""
    beta = np.polyfit(x.values, y.values, 1)[0]
    return float(beta)


@dataclass
class PairSignalResult:
    df: pd.DataFrame
    pairs: List[Pair]


def generate_signals(
    df: pd.DataFrame,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    significance: float = 0.05,
) -> PairSignalResult:
    """Detect cointegrated pairs and compute trading signals.

    Returns a :class:`PairSignalResult` containing the enriched dataframe and
    the list of detected pairs.  The dataframe includes z-score columns named
    ``pair_z_<SYM1>_<SYM2>`` together with ``pair_long``/``pair_short`` entry
    signals and cumulative ``pair_pnl`` based on the spread return.
    """

    df = df.sort_values("Timestamp").reset_index(drop=True)
    pairs = find_cointegrated_pairs(df, significance=significance)
    if not pairs:
        out = df.copy()
        out["pair_long"] = 0
        out["pair_short"] = 0
        out["pair_pnl"] = 0.0
        return PairSignalResult(out, [])

    wide = df.pivot(index="Timestamp", columns="Symbol", values="Bid").sort_index()
    long_sig = pd.Series(0, index=wide.index)
    short_sig = pd.Series(0, index=wide.index)
    pnl = pd.Series(0.0, index=wide.index)

    for sym1, sym2, _ in pairs:
        y = wide[sym1]
        x = wide[sym2]
        beta = _hedge_ratio(y, x)
        spread = y - beta * x
        z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
        zname = f"pair_z_{sym1}_{sym2}"
        df[zname] = z.reindex(df["Timestamp"]).values
        df[f"hedge_{sym1}_{sym2}"] = beta

        position = 0
        prev_spread = spread.shift(1)
        for t in range(len(spread)):
            z_t = z.iloc[t]
            if np.isnan(z_t):
                continue
            if position == 0:
                if z_t < -entry_z:
                    long_sig.iloc[t] = 1
                    position = 1
                elif z_t > entry_z:
                    short_sig.iloc[t] = 1
                    position = -1
            elif position == 1 and z_t > -exit_z:
                position = 0
            elif position == -1 and z_t < exit_z:
                position = 0
            if t > 0 and not np.isnan(prev_spread.iloc[t]) and not np.isnan(spread.iloc[t]):
                pnl.iloc[t] += position * (spread.iloc[t] - prev_spread.iloc[t])

    df["pair_long"] = long_sig.reindex(df["Timestamp"]).fillna(0).astype(int).values
    df["pair_short"] = short_sig.reindex(df["Timestamp"]).fillna(0).astype(int).values
    df["pair_pnl"] = pnl.reindex(df["Timestamp"]).fillna(0.0).values
    return PairSignalResult(df, pairs)


def signal_from_features(features: Dict[str, float]) -> float:
    """Router helper returning action from feature dict."""
    return float(features.get("pair_long", 0.0) - features.get("pair_short", 0.0))


__all__ = [
    "find_cointegrated_pairs",
    "generate_signals",
    "signal_from_features",
    "PairSignalResult",
]
