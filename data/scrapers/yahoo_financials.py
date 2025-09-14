"""Scraper for Yahoo Finance fundamental metrics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

import pandas as pd

from . import fetch_with_cache

URL = (
    "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?"
    "modules=financialData,defaultKeyStatistics"
)


def parse(symbol: str, raw: str) -> pd.DataFrame:
    """Parse Yahoo Finance ``quoteSummary`` JSON for ``symbol``.

    Parameters
    ----------
    symbol:
        Ticker symbol used in the query.
    raw:
        Raw JSON string returned by the Yahoo Finance endpoint.

    Returns
    -------
    pd.DataFrame
        Single-row dataframe with a ``Date`` column and fundamental metrics.
    """

    data = json.loads(raw)
    result = data.get("quoteSummary", {}).get("result", [{}])[0]
    financial = result.get("financialData", {})
    stats = result.get("defaultKeyStatistics", {})
    row = {
        "Date": pd.Timestamp.utcnow().normalize(),
        "eps": stats.get("forwardEps", {}).get("raw"),
        "revenue": financial.get("totalRevenue", {}).get("raw"),
        "ebitda": financial.get("ebitda", {}).get("raw"),
        "market_cap": stats.get("marketCap", {}).get("raw"),
    }
    return pd.DataFrame([row])


async def fetch(symbol: str, force_refresh: bool = False, client=None) -> pd.DataFrame:
    """Fetch fundamentals for ``symbol`` and cache to ``data/fundamentals``."""

    raw = await fetch_with_cache(
        URL.format(symbol=symbol), f"yahoo_{symbol}.json", client, force_refresh
    )
    df = parse(symbol, raw)
    out_dir = Path("data") / "fundamentals"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{symbol}.csv", index=False)
    return df


async def download(symbols: List[str], force_refresh: bool = False) -> None:
    """Download fundamentals for multiple ``symbols`` concurrently."""

    await asyncio.gather(
        *(fetch(sym, force_refresh=force_refresh) for sym in symbols)
    )


async def main(force_refresh: bool = False, symbols: List[str] | None = None) -> None:
    if symbols is None:
        symbols = ["AAPL"]
    await download(symbols, force_refresh=force_refresh)
    print(f"Fetched fundamentals for {symbols}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Yahoo Finance fundamentals scraper")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols to fetch")
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cache and refetch"
    )
    args = parser.parse_args()
    asyncio.run(main(force_refresh=args.refresh, symbols=args.symbols))
