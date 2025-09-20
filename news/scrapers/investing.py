"""Scraper for Investing.com news RSS feed."""

from __future__ import annotations

import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Dict, List
from xml.etree import ElementTree as ET

URL = "https://www.investing.com/rss/news_25.rss"

_PAREN_SYMBOL_RE = re.compile(r"\(([A-Z]{1,5})\)")
_DOLLAR_SYMBOL_RE = re.compile(r"\$([A-Z]{1,5})\b")


def _clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_symbols(text: str) -> List[str]:
    symbols: List[str] = []
    for pattern in (_PAREN_SYMBOL_RE, _DOLLAR_SYMBOL_RE):
        for match in pattern.findall(text):
            sym = match.strip().upper()
            if sym not in symbols:
                symbols.append(sym)
    return symbols


def parse(text: str) -> List[Dict]:
    items: List[Dict] = []
    if not text:
        return items
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return items
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        timestamp = _parse_timestamp(node.findtext("pubDate"))
        if not (title and link and timestamp):
            continue
        description = _clean_html(node.findtext("description") or "")
        combined = " ".join([title, description])
        symbols = _extract_symbols(combined)
        items.append(
            {
                "timestamp": timestamp,
                "title": title,
                "url": link,
                "summary": description,
                "symbols": symbols,
                "source": "Investing.com",
            }
        )
    return items


async def fetch() -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, timeout=10) as resp:
            text = await resp.text()
    return parse(text)


__all__ = ["parse", "fetch", "URL"]

