"""Load and merge basic fundamental metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def _read_local_csv(symbol: str) -> pd.DataFrame:
    """Return fundamental data for ``symbol`` from dataset or data folders.

    The function looks for ``<symbol>.csv`` under ``dataset/fundamentals`` and
    ``data/fundamentals``. Files are expected to contain at least a ``Date``
    column and one value column (e.g. ``pe_ratio``). Missing files result in an
    empty dataframe.
    """

    paths = [
        Path("dataset") / "fundamentals" / f"{symbol}.csv",
        Path("data") / "fundamentals" / f"{symbol}.csv",
    ]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - local read failure
                logger.warning(
                    "Failed to read fundamentals for %s from %s", symbol, path
                )
                return pd.DataFrame()
    return pd.DataFrame()


def _fetch_yfinance(symbol: str) -> pd.DataFrame:
    """Attempt to fetch fundamental data via ``yfinance``.

    A small subset of commonly used metrics such as ``eps`` and ``ebitda`` is
    retrieved and written to ``data/fundamentals/<symbol>.csv`` so subsequent
    runs can operate offline.  Network failures simply return an empty
    dataframe.
    """

    try:  # pragma: no cover - network and dependency optional
        import yfinance as yf  # type: ignore

        ticker = yf.Ticker(symbol)
        info = getattr(ticker, "info", {})
        df = pd.DataFrame(
            {
                "Date": pd.Timestamp.utcnow().normalize(),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "eps": info.get("forwardEps"),
                "revenue": info.get("totalRevenue"),
                "ebitda": info.get("ebitda"),
                "market_cap": info.get("marketCap"),
            },
            index=[0],
        )
        # Persist to disk so future runs can operate offline
        out_dir = Path("data") / "fundamentals"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:  # pragma: no cover - file system edge cases
            df.to_csv(out_dir / f"{symbol}.csv", index=False)
        except Exception:
            logger.warning("Failed to cache fundamentals for %s", symbol)
        return df
    except Exception:
        logger.debug("yfinance unavailable for %s", symbol)
        return pd.DataFrame()


def load_fundamentals(symbols: Iterable[str]) -> pd.DataFrame:
    """Load fundamental metrics for ``symbols``.

    Parameters
    ----------
    symbols:
        Collection of ticker symbols.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Date``, ``Symbol`` and fundamental metrics. If no
        data is available an empty dataframe with the expected columns is
        returned.
    """

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = _read_local_csv(sym)
        if df.empty:
            df = _fetch_yfinance(sym)
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Symbol"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Date",
                "Symbol",
                "pe_ratio",
                "dividend_yield",
                "eps",
                "revenue",
                "ebitda",
                "market_cap",
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Symbol", "Date"])


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Merge fundamental metrics into ``df``.

    The resulting dataframe contains ``pe_ratio`` and ``dividend_yield`` aligned
    with price ``Timestamp`` on a per-symbol basis using a backward as-of join.
    Missing values are forward-filled then set to zero.
    """

    if "Symbol" not in df.columns:
        # Nothing to merge; return zeros for expected columns
        for col in [
            "pe_ratio",
            "dividend_yield",
            "eps",
            "revenue",
            "ebitda",
            "market_cap",
        ]:
            df[col] = 0.0
        return df

    fundamentals = load_fundamentals(sorted(df["Symbol"].unique()))
    if fundamentals.empty:
        for col in [
            "pe_ratio",
            "dividend_yield",
            "eps",
            "revenue",
            "ebitda",
            "market_cap",
        ]:
            df[col] = 0.0
        return df

    fundamentals = fundamentals.rename(columns={"Date": "fund_date"})
    df = pd.merge_asof(
        df.sort_values("Timestamp"),
        fundamentals.sort_values("fund_date"),
        left_on="Timestamp",
        right_on="fund_date",
        by="Symbol",
        direction="backward",
    ).drop(columns=["fund_date"])

    for col in [
        "pe_ratio",
        "dividend_yield",
        "eps",
        "revenue",
        "ebitda",
        "market_cap",
    ]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)
        else:
            df[col] = 0.0
    return df


__all__ = ["load_fundamentals", "compute"]
