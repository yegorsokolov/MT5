import csv
import json
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Callable, Awaitable

import requests
from bs4 import BeautifulSoup

from .scrapers import forexfactory_news, cbc, cnn, reuters, yahoo


DEFAULT_SCRAPERS = [
    forexfactory_news.fetch,
    cbc.fetch,
    cnn.fetch,
    reuters.fetch,
    yahoo.fetch,
]


def _normalise_importance(value: Optional[str]) -> Optional[str]:
    """Return a colour-coded importance level."""

    if not value:
        return value
    val = str(value).strip().lower()
    mapping = {
        "high": "red",
        "medium": "orange",
        "low": "yellow",
    }
    return mapping.get(val, val)


class NewsAggregator:
    """Aggregate economic and headline news from multiple sources.

    Events are cached on disk in JSON format.  Historical behaviour for the
    economic calendar aggregation is preserved for backwards compatibility.
    Additional functionality is provided for aggregating headline news across
    a variety of scrapers.
    """

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/news_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "news.json"
        self.ttl = timedelta(hours=ttl_hours)

    # ------------------------------------------------------------------
    # Cache handling
    def _load_cache(self) -> List[Dict]:
        if not self.cache_file.exists():
            return []
        with self.cache_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        events: List[Dict] = []
        for ev in raw:
            ev = ev.copy()
            if isinstance(ev.get("timestamp"), str):
                ev["timestamp"] = datetime.fromisoformat(ev["timestamp"])
            events.append(ev)
        return events

    def _save_cache(self, events: List[Dict]) -> None:
        serialisable: List[Dict] = []
        for ev in events:
            ev = ev.copy()
            ts = ev.get("timestamp")
            if isinstance(ts, datetime):
                ev["timestamp"] = ts.isoformat()
            serialisable.append(ev)
        with self.cache_file.open("w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)

    def _apply_ttl(self, events: List[Dict]) -> List[Dict]:
        cutoff = datetime.now(timezone.utc) - self.ttl
        return [ev for ev in events if ev.get("timestamp") and ev["timestamp"] >= cutoff]

    # ------------------------------------------------------------------
    def get_news(self, start: datetime, end: datetime) -> List[Dict]:
        """Return cached news events within the inclusive [start, end] range."""
        events = self._load_cache()
        return [ev for ev in events if start <= ev["timestamp"] <= end]

    def get_symbol_news(self, symbol: str, start: datetime, end: datetime) -> List[Dict]:
        events = self._load_cache()
        results: List[Dict] = []
        for ev in events:
            ts = ev.get("timestamp")
            if not ts:
                continue
            if start <= ts <= end and symbol in ev.get("symbols", []):
                results.append(ev)
        return results

    # ------------------------------------------------------------------
    # Fetching logic
    def _fetch(self, url: str) -> Optional[str]:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, timeout=10, headers=headers)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None

    def fetch(self) -> List[Dict]:
        """Fetch, parse and cache events from all known sources."""
        sources = [
            (
                "faireconomy_xml",
                "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml",
                self.parse_faireconomy_xml,
            ),
            (
                "faireconomy_csv",
                "https://nfs.faireconomy.media/ff_calendar_thisweek.csv",
                self.parse_faireconomy_csv,
            ),
            (
                "forexfactory_html",
                "https://www.forexfactory.com/calendar",
                self.parse_forexfactory_html,
            ),
            (
                "tradingeconomics_json",
                "https://api.tradingeconomics.com/calendar?c=guest:guest&format=json",
                self.parse_tradingeconomics_json,
            ),
        ]
        events = self._load_cache()
        for name, url, parser in sources:
            text = self._fetch(url)
            if not text:
                continue
            try:
                parsed = parser(text, source=name)
                events.extend(parsed)
            except Exception:
                continue
        events = self._dedupe(events)
        self._save_cache(events)
        return events

    async def refresh(self, scrapers: Optional[List[Callable[[], Awaitable[List[Dict]]]]] = None) -> List[Dict]:
        """Fetch headline news from scrapers asynchronously and update cache."""
        scrapers = scrapers or DEFAULT_SCRAPERS
        existing = self._load_cache()
        results = await asyncio.gather(*[scraper() for scraper in scrapers], return_exceptions=True)
        new_events: List[Dict] = []
        for res in results:
            if isinstance(res, list):
                new_events.extend(res)
        combined = self._dedupe_news(existing + new_events)
        combined = self._apply_ttl(combined)
        self._save_cache(combined)
        return combined

    # ------------------------------------------------------------------
    # Parsing helpers
    def parse_faireconomy_xml(self, text: str, source: str = "faireconomy_xml") -> List[Dict]:
        from xml.etree import ElementTree as ET

        events: List[Dict] = []
        root = ET.fromstring(text)
        for ev in root.findall(".//event"):
            try:
                eid = ev.get("id")
                title = ev.findtext("title") or ev.findtext("event")
                currency = ev.findtext("currency") or ev.findtext("country")
                date_str = ev.findtext("date")
                time_str = ev.findtext("time") or "00:00"
                ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                impact = ev.findtext("impact")
                forecast = ev.findtext("forecast")
                actual = ev.findtext("actual")
                events.append(
                    {
                        "id": eid,
                        "timestamp": ts,
                        "currency": currency,
                        "currencies": [currency] if currency else [],
                        "event": title,
                        "actual": actual,
                        "forecast": forecast,
                        "importance": _normalise_importance(impact),
                        "symbols": [],
                        "sources": [source],
                    }
                )
            except Exception:
                continue
        return events

    def parse_faireconomy_csv(self, text: str, source: str = "faireconomy_csv") -> List[Dict]:
        events: List[Dict] = []
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            try:
                eid = row.get("id") or row.get("eventid")
                title = row.get("title") or row.get("event")
                currency = row.get("currency") or row.get("country")
                date_str = row.get("date")
                time_str = row.get("time") or "00:00"
                ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                events.append(
                    {
                        "id": eid,
                        "timestamp": ts,
                        "currency": currency,
                        "currencies": [currency] if currency else [],
                        "event": title,
                        "actual": row.get("actual"),
                        "forecast": row.get("forecast"),
                        "importance": _normalise_importance(row.get("impact")),
                        "symbols": [],
                        "sources": [source],
                    }
                )
            except Exception:
                continue
        return events

    def parse_forexfactory_html(self, text: str, source: str = "forexfactory_html") -> List[Dict]:
        soup = BeautifulSoup(text, "html.parser")
        events: List[Dict] = []
        for row in soup.select("tr[data-event-id]"):
            try:
                eid = row.get("data-event-id")
                ts_val = row.get("data-timestamp")
                if ts_val:
                    ts = datetime.fromtimestamp(int(ts_val), tz=timezone.utc)
                else:
                    ts = None
                currency_cell = row.find(class_="calendar__currency")
                event_cell = row.find(class_="calendar__event")
                actual_cell = row.find(class_="calendar__actual")
                forecast_cell = row.find(class_="calendar__forecast")
                importance = row.get("data-impact")
                cur = currency_cell.get_text(strip=True) if currency_cell else None
                events.append(
                    {
                        "id": eid,
                        "timestamp": ts,
                        "currency": cur,
                        "currencies": [cur] if cur else [],
                        "event": event_cell.get_text(strip=True) if event_cell else None,
                        "actual": actual_cell.get_text(strip=True) if actual_cell else None,
                        "forecast": forecast_cell.get_text(strip=True) if forecast_cell else None,
                        "importance": _normalise_importance(importance),
                        "symbols": [],
                        "sources": [source],
                    }
                )
            except Exception:
                continue
        return events

    def parse_tradingeconomics_json(
        self, text: str, source: str = "tradingeconomics_json"
    ) -> List[Dict]:
        """Parse TradingEconomics calendar JSON."""

        events: List[Dict] = []
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return events

        for item in items:
            try:
                date_str = item.get("date") or item.get("Date")
                event_name = item.get("event") or item.get("Event")
                if not (date_str and event_name):
                    continue
                try:
                    ts = datetime.fromisoformat(date_str.replace("Z", "")).replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    continue
                currency = item.get("currency") or item.get("Country")
                events.append(
                    {
                        "id": item.get("id"),
                        "timestamp": ts,
                        "currency": currency,
                        "currencies": [currency] if currency else [],
                        "event": event_name,
                        "actual": item.get("actual") or item.get("Actual"),
                        "forecast": item.get("forecast") or item.get("Forecast"),
                        "importance": _normalise_importance(
                            item.get("importance") or item.get("Importance")
                        ),
                        "symbols": [],
                        "sources": [source],
                    }
                )
            except Exception:
                continue
        return events

    # ------------------------------------------------------------------
    def _dedupe(self, events: List[Dict]) -> List[Dict]:
        """Deduplicate events by (id, timestamp, currency, event)."""
        deduped: Dict[tuple, Dict] = {}
        for ev in events:
            key = (
                ev.get("id"),
                ev.get("timestamp"),
                ev.get("currency"),
                ev.get("event"),
            )
            existing = deduped.get(key)
            if existing:
                # merge sources
                existing_sources = set(existing.get("sources", []))
                new_sources = set(ev.get("sources", []))
                existing["sources"] = list(existing_sources | new_sources)
                # fill missing fields
                for field in ["actual", "forecast", "importance"]:
                    if not existing.get(field) and ev.get(field):
                        existing[field] = ev[field]
                for field in ["currencies", "symbols"]:
                    if ev.get(field):
                        merged = set(existing.get(field, [])) | set(ev.get(field, []))
                        existing[field] = list(merged)
            else:
                deduped[key] = ev
        return list(deduped.values())

    def _dedupe_news(self, events: List[Dict]) -> List[Dict]:
        """Deduplicate headline news by (symbol, title, timestamp)."""
        deduped: Dict[tuple, Dict] = {}
        for ev in events:
            symbols = ev.get("symbols", []) or []
            for sym in symbols:
                key = (sym, ev.get("title"), ev.get("timestamp"))
                existing = deduped.get(key)
                if existing:
                    merged = set(existing.get("symbols", [])) | set(symbols)
                    existing["symbols"] = list(merged)
                else:
                    deduped[key] = ev
        return list(deduped.values())


__all__ = ["NewsAggregator"]
