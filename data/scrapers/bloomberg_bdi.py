"""Scraper for Bloomberg's Baltic Dry Index quote (BDIY:IND)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # Python < 3.11
    UTC = timezone.utc  # type: ignore[misc]
from typing import List, Dict

from bs4 import BeautifulSoup

from . import fetch_with_cache

URL = "https://www.bloomberg.com/quote/BDIY:IND"


def _extract_price(html: str) -> float:
    soup = BeautifulSoup(html, "html.parser")
    # Bloomberg uses dynamic classes; try a few selectors
    tag = soup.select_one('[data-testid="price"]') or soup.find(class_="price")
    if tag:
        text = tag.get_text(strip=True)
    else:
        # Fallback: first number in page
        match = re.search(r"[0-9,.]+", soup.get_text())
        if not match:
            raise ValueError("Price not found")
        text = match.group(0)
    return float(text.replace(",", ""))


def parse(html: str) -> List[Dict[str, object]]:
    price = _extract_price(html)
    return [
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": "BDIY:IND",
            "value": price,
            "source": "Bloomberg",
        }
    ]


async def fetch(force_refresh: bool = False, client=None) -> List[Dict[str, object]]:
    raw = await fetch_with_cache(URL, "bloomberg_bdi.html", client, force_refresh)
    return parse(raw)


async def main(force_refresh: bool = False) -> None:
    data = await fetch(force_refresh=force_refresh)
    import json as _json

    print(_json.dumps(data, indent=2))


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Bloomberg BDIY scraper")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and refetch")
    args = parser.parse_args()
    asyncio.run(main(force_refresh=args.refresh))
