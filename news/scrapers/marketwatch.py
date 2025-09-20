"""Scraper for MarketWatch top stories RSS feed."""

from __future__ import annotations

import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Dict, List
from xml.etree import ElementTree as ET

URL = "https://feeds.marketwatch.com/marketwatch/topstories/"

_TEXT_SYMBOL_RE = re.compile(r"\$([A-Z]{1,5})|\(([A-Z]{1,5})\)")


def _clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


def _unique_symbols(symbols: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        ordered.append(sym)
    return ordered


def _extract_symbols(item: ET.Element, fallback_text: str) -> List[str]:
    symbols: List[str] = []
    for node in item.findall(".//{*}ticker"):
        if node.text:
            cleaned = node.text.strip().lstrip("$").upper()
            if cleaned:
                symbols.append(cleaned)
    fallback_text = fallback_text or ""
    for match in _TEXT_SYMBOL_RE.finditer(fallback_text):
        sym = match.group(1) or match.group(2)
        if sym:
            symbols.append(sym.upper())
    return _unique_symbols(symbols)


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
        symbol_text = " ".join([description, title])
        symbols = _extract_symbols(node, symbol_text)
        items.append(
            {
                "timestamp": timestamp,
                "title": title,
                "url": link,
                "summary": description,
                "symbols": symbols,
                "source": "MarketWatch",
            }
        )
    return items


async def fetch() -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(URL, timeout=10) as resp:
            text = await resp.text()
    return parse(text)


__all__ = ["parse", "fetch", "URL"]

