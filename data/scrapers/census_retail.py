"""Scraper for U.S. Census monthly retail sales."""

from __future__ import annotations

import csv
from io import StringIO
from typing import List, Dict

from . import fetch_with_cache

URL = "https://www.census.gov/retail/marts/www/marts_current_release.csv"


def parse(csv_text: str) -> List[Dict[str, object]]:
    reader = csv.DictReader(StringIO(csv_text))
    records: List[Dict[str, object]] = []
    for row in reader:
        ts = row.get("Month") or row.get("Date")
        val = row.get("Sales") or row.get("Value")
        if not (ts and val):
            continue
        try:
            value = float(val.replace(",", ""))
        except ValueError:
            continue
        records.append(
            {
                "timestamp": ts,
                "symbol": "US_RETAIL_SALES",
                "value": value,
                "source": "US Census",
            }
        )
    return records


async def fetch(force_refresh: bool = False, client=None) -> List[Dict[str, object]]:
    raw = await fetch_with_cache(URL, "census_retail.csv", client, force_refresh)
    return parse(raw)


async def main(force_refresh: bool = False) -> None:
    data = await fetch(force_refresh=force_refresh)
    import json as _json

    print(_json.dumps(data, indent=2))


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Census retail sales scraper")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and refetch")
    args = parser.parse_args()
    asyncio.run(main(force_refresh=args.refresh))
