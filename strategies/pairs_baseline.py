"""Cross-sectional relative strength pair trading baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class PairsBaselineConfig:
    """Configuration for :func:`generate_signals`.

    Parameters
    ----------
    universe:
        Iterable of symbols to consider.  When ``None`` all symbols
        present in the dataframe are used.
    hedge_ratio:
        Size of the short leg relative to the long leg.  A value of ``1``
        results in a market-neutral position.
    """

    universe: Optional[Iterable[str]] = None
    hedge_ratio: float = 1.0


def generate_signals(df: pd.DataFrame, config: Optional[PairsBaselineConfig] = None) -> pd.DataFrame:
    """Return dataframe with long/short signals based on relative strength.

    The function expects ``df`` to contain cross-sectional relative
    strength features named ``rel_strength_<sym>`` for each symbol.  For
    every timestamp the symbol with the highest relative strength is
    assigned a long signal of ``+1`` while the weakest symbol receives a
    short signal scaled by ``hedge_ratio``.  All other rows remain ``0``.
    """

    if config is None:
        config = PairsBaselineConfig()

    universe = list(config.universe) if config.universe is not None else sorted(df["Symbol"].unique())

    df = df.copy().sort_values("Timestamp")
    df["signal"] = 0.0

    groups = df[df["Symbol"].isin(universe)].groupby("Timestamp")
    for ts, grp in groups:
        strengths: dict[str, float] = {}
        for sym in universe:
            col = f"rel_strength_{sym}"
            row = grp[grp["Symbol"] == sym]
            if row.empty or col not in df.columns:
                continue
            strengths[sym] = float(row.iloc[0][col])
        if len(strengths) < 2:
            continue
        long_sym = max(strengths, key=strengths.get)
        short_sym = min(strengths, key=strengths.get)
        if long_sym == short_sym:
            continue
        df.loc[(df["Timestamp"] == ts) & (df["Symbol"] == long_sym), "signal"] = 1.0
        df.loc[(df["Timestamp"] == ts) & (df["Symbol"] == short_sym), "signal"] = -float(config.hedge_ratio)

    return df


__all__ = ["PairsBaselineConfig", "generate_signals"]


def run_backtest(
    cfg: dict,
    *,
    latency_ms: int = 0,
    slippage_model=None,
    model=None,
):
    """Convenience wrapper to backtest with execution settings."""
from mt5.backtest import run_backtest as _run_backtest

    return _run_backtest(
        cfg,
        latency_ms=latency_ms,
        slippage_model=slippage_model,
        model=model,
    )


__all__.append("run_backtest")
