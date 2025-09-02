"""Scraper for TradingEconomics economic calendar."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import List, Dict

from . import fetch_with_cache

URL = "https://api.tradingeconomics.com/calendar?c=all&format=json"


def _parse_value(val: str | float | int | None) -> float | None:
    """Convert TradingEconomics value strings to float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = re.sub(r"[^0-9\.-]", "", val)
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse(raw: str) -> List[Dict[str, object]]:
    """Parse raw JSON from TradingEconomics calendar."""
    try:
        events = json.loads(raw)
    except json.JSONDecodeError:
        return []

    records: List[Dict[str, object]] = []
    for item in events:
        timestamp = item.get("date") or item.get("Date")
        symbol = item.get("event") or item.get("Event")
        value = _parse_value(item.get("actual") or item.get("Actual"))
        if not (timestamp and symbol and value is not None):
            continue
        # Normalise timestamp to ISO format if possible
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "")).isoformat()
        except ValueError:
            pass
        records.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "value": value,
                "source": "TradingEconomics",
            }
        )
    return records


async def fetch(force_refresh: bool = False, client=None) -> List[Dict[str, object]]:
    """Fetch and parse the TradingEconomics calendar."""
    raw = await fetch_with_cache(URL, "tradingeconomics_calendar.json", client, force_refresh)
    return parse(raw)


async def main(force_refresh: bool = False) -> None:
    data = await fetch(force_refresh=force_refresh)
    import json as _json

    print(_json.dumps(data, indent=2))


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="TradingEconomics calendar scraper")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and refetch")
    args = parser.parse_args()
    asyncio.run(main(force_refresh=args.refresh))
