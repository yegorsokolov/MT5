"""General purpose scraper for broad market news feeds.

The scraper expects a JSON payload with an ``items`` key that contains a
collection of articles.  Each item should provide ``title`` and
``published_at`` (or ``published``/``time``) fields alongside optional
``summary`` and ``tickers`` information.  The parser normalises these entries
into the common headline structure used throughout the news package.

The module purposefully keeps its heuristics lightweight so that it can run in
restricted environments (e.g. tests) while still supporting richer payloads
served by external providers when network access is available.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import unescape
from typing import Dict, Iterable, List

import aiohttp

URL = "https://raw.githubusercontent.com/telegeography/data/master/news/sample.json"

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str | None) -> str:
    """Return ``text`` stripped of HTML markup and normalised whitespace."""

    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = unescape(cleaned)
    return " ".join(cleaned.split())


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse common timestamp formats into UTC aware :class:`datetime`."""

    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            dt = datetime.strptime(value.replace("Z", "+00:00"), fmt)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except Exception:
        return None


def _normalise_tickers(tickers: Iterable[str] | str | None) -> List[str]:
    if tickers is None:
        return []
    if isinstance(tickers, str):
        tickers = tickers.split(",")
    normalised: List[str] = []
    for ticker in tickers:
        if not ticker:
            continue
        t = str(ticker).strip().upper()
        if t:
            normalised.append(t)
    return normalised


def parse(text: str) -> List[Dict]:
    """Parse a JSON payload into a list of headline dictionaries."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    items = data.get("items") or data.get("articles") or []
    results: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ts = _parse_timestamp(
            item.get("published_at")
            or item.get("published")
            or item.get("time")
            or item.get("date")
        )
        if not ts:
            continue
        title = _clean_text(item.get("title") or item.get("headline"))
        url = item.get("url") or item.get("link") or ""
        summary = _clean_text(item.get("summary") or item.get("description"))
        symbols = _normalise_tickers(
            item.get("tickers") or item.get("symbols") or item.get("related"),
        )
        results.append(
            {
                "timestamp": ts,
                "title": title,
                "url": url,
                "summary": summary,
                "symbols": symbols,
                "source": item.get("source") or item.get("publisher"),
            }
        )
    return results


async def fetch() -> List[Dict]:
    """Fetch the remote feed returning parsed headlines.

    Any network or parsing error results in an empty list, allowing the caller
    to degrade gracefully without disrupting other scrapers.
    """

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(URL, timeout=10) as resp:
                text = await resp.text()
    except Exception:
        return []
    return parse(text)


__all__ = ["fetch", "parse"]

