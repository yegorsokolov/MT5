import csv
import json
import asyncio
from datetime import datetime, timezone, timedelta
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:  # pragma: no cover - optional dependency in lightweight environments
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from bs4 import BeautifulSoup

try:  # pragma: no cover - pandas is optional at runtime
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .scrapers import (
    cbc,
    cnn,
    forexfactory_news,
    global_feed,
    investing,
    marketwatch,
    reuters,
    yahoo,
)

try:  # pragma: no cover - optional AI helpers
    from . import ai_enrichment
except Exception:  # pragma: no cover
    ai_enrichment = None  # type: ignore


DEFAULT_SCRAPERS = [
    forexfactory_news.fetch,
    cbc.fetch,
    cnn.fetch,
    reuters.fetch,
    yahoo.fetch,
    global_feed.fetch,
    marketwatch.fetch,
    investing.fetch,
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


_POSITIVE_TERMS = {
    "beat",
    "beats",
    "surge",
    "surges",
    "rally",
    "gain",
    "gains",
    "up",
    "record",
    "growth",
    "strong",
    "optimism",
    "soar",
    "soars",
    "improve",
    "improves",
    "bullish",
    "expand",
    "expands",
}

_NEGATIVE_TERMS = {
    "fall",
    "falls",
    "drop",
    "drops",
    "down",
    "slump",
    "loss",
    "losses",
    "warning",
    "weak",
    "decline",
    "declines",
    "plunge",
    "plunges",
    "miss",
    "misses",
    "bearish",
    "shrink",
    "shrinks",
}


def _headline_sentiment(text: str) -> float:
    """Return a heuristic sentiment score in the ``[-1, 1]`` range."""

    if not text:
        return 0.0
    words = [w.strip(".,!?;:""'()[]{}<>\n\r").lower() for w in text.split()]
    pos = sum(1 for w in words if w in _POSITIVE_TERMS)
    neg = sum(1 for w in words if w in _NEGATIVE_TERMS)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(pos + neg, 1)
    return max(-1.0, min(1.0, float(score)))


def _ensure_datetime(value: Any) -> Optional[datetime]:
    """Coerce ``value`` into an aware :class:`datetime` when possible."""

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


_IMPACT_MODEL: ModuleType | None = None


def _load_impact_model() -> Optional[ModuleType]:
    """Lazily import :mod:`news.impact_model` returning ``None`` on failure."""

    global _IMPACT_MODEL
    if _IMPACT_MODEL is None:
        try:
            _IMPACT_MODEL = import_module("news.impact_model")
        except Exception:
            _IMPACT_MODEL = None
    return _IMPACT_MODEL


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

    def _analyse_headlines(self, events: List[Dict]) -> List[Dict]:
        """Attach heuristic sentiment and impact estimates to ``events``."""

        if not events:
            return []

        analysed: List[Dict] = []
        rows: List[Dict[str, Any]] = []
        for idx, event in enumerate(events):
            ev = dict(event)
            ts = _ensure_datetime(ev.get("timestamp"))
            if ts:
                ev["timestamp"] = ts

            text = " ".join(
                part.strip() for part in [ev.get("title", ""), ev.get("summary", "")] if part
            )
            sentiment = ev.get("sentiment")
            if sentiment is None:
                sentiment = _headline_sentiment(text)
            else:
                try:
                    sentiment = float(sentiment)
                except (TypeError, ValueError):
                    sentiment = _headline_sentiment(text)
            sentiment = float(sentiment)
            ev["sentiment"] = sentiment

            analysis = dict(ev.get("analysis") or {})
            analysis["sentiment"] = sentiment

            if ai_enrichment is not None:
                summary_source = ev.get("summary") or text
                ai_summary = ai_enrichment.summarize_text(summary_source)
                if ai_summary:
                    analysis["summary"] = ai_summary

                keywords = ai_enrichment.extract_keywords(text)
                if keywords:
                    analysis["keywords"] = keywords

                topics = ai_enrichment.classify_topics(text)
                if topics:
                    analysis["topics"] = topics

                ml_sentiment = ai_enrichment.predict_sentiment(text)
                if ml_sentiment is not None:
                    analysis["ml_sentiment"] = ml_sentiment
                    if sentiment:
                        sentiment = float((sentiment + ml_sentiment) / 2)
                    else:
                        sentiment = float(ml_sentiment)
                    sentiment = max(-1.0, min(1.0, sentiment))
                    ev["sentiment"] = sentiment
                    analysis["sentiment"] = sentiment

            impact = ev.get("impact")
            if impact is not None:
                try:
                    analysis["impact"] = float(impact)
                    ev["impact"] = float(impact)
                except (TypeError, ValueError):
                    ev.pop("impact", None)
            uncertainty = ev.get("impact_uncertainty")
            if uncertainty is not None:
                try:
                    analysis["uncertainty"] = float(uncertainty)
                    ev["impact_uncertainty"] = float(uncertainty)
                except (TypeError, ValueError):
                    ev.pop("impact_uncertainty", None)

            symbols = ev.get("symbols") or []
            if not isinstance(symbols, list):
                symbols = list(symbols) if symbols else []
            cleaned_symbols: List[str] = []
            for sym in symbols:
                if not sym:
                    continue
                cleaned = str(sym).strip().upper()
                if cleaned:
                    cleaned_symbols.append(cleaned)
            ev["symbols"] = cleaned_symbols
            analysed.append(ev)

            if cleaned_symbols and isinstance(ev.get("timestamp"), datetime):
                for sym in cleaned_symbols:
                    rows.append(
                        {
                            "event_idx": idx,
                            "symbol": sym,
                            "timestamp": ev["timestamp"],
                            "sentiment": sentiment,
                            "surprise": 0.0,
                            "historical_response": 0.0,
                            "text": text or ev.get("title", ""),
                        }
                    )

            ev["analysis"] = analysis

        model = _load_impact_model()
        if rows and pd is not None and model is not None and hasattr(model, "score"):
            df = pd.DataFrame(rows)
            try:
                score_df = model.score(
                    df[[
                        "symbol",
                        "timestamp",
                        "surprise",
                        "sentiment",
                        "historical_response",
                        "text",
                    ]]
                )
            except Exception:
                score_df = None

            if score_df is not None and not score_df.empty:
                df = df.reset_index(drop=True)
                score_df = score_df.reset_index(drop=True)
                df["impact"] = score_df["impact"].astype(float)
                if "uncertainty" in score_df.columns:
                    df["uncertainty"] = score_df["uncertainty"].astype(float)
                else:
                    df["uncertainty"] = 0.0
                for event_idx, grp in df.groupby("event_idx"):
                    impact_val = float(grp["impact"].mean())
                    uncert_val = float(grp["uncertainty"].mean())
                    analysed[event_idx]["impact"] = impact_val
                    analysed[event_idx]["impact_uncertainty"] = uncert_val
                    analysis = dict(analysed[event_idx].get("analysis") or {})
                    analysis["impact"] = impact_val
                    analysis["uncertainty"] = uncert_val
                    analysed[event_idx]["analysis"] = analysis

        event_sentiments: Dict[int, List[float]] = {}
        for row in rows:
            event_sentiments.setdefault(row["event_idx"], []).append(float(row["sentiment"]))
        for idx, sentiments in event_sentiments.items():
            if analysed[idx].get("impact") is not None:
                continue
            if sentiments:
                impact_val = float(sum(sentiments) / len(sentiments))
                analysed[idx]["impact"] = impact_val
                analysed[idx]["impact_uncertainty"] = analysed[idx].get(
                    "impact_uncertainty", 1.0
                )
                analysis = dict(analysed[idx].get("analysis") or {})
                analysis.setdefault("impact", impact_val)
                analysis.setdefault("uncertainty", analysed[idx]["impact_uncertainty"])
                analysed[idx]["analysis"] = analysis

        return analysed

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
        if requests is None:  # pragma: no cover - exercised when requests missing
            return None
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
        existing = self._analyse_headlines(self._load_cache())
        results = await asyncio.gather(*[scraper() for scraper in scrapers], return_exceptions=True)
        new_events: List[Dict] = []
        for res in results:
            if isinstance(res, list):
                new_events.extend(res)
        analysed_new = self._analyse_headlines(new_events)
        combined = self._dedupe_news(existing + analysed_new)
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
        unique: List[Dict] = []
        for ev in events:
            symbols = ev.get("symbols") or []
            if not isinstance(symbols, list):
                symbols = list(symbols)
            keys = symbols if symbols else [None]
            added = False
            created_key = False
            for sym in keys:
                key = (sym, ev.get("title"), ev.get("timestamp"))
                existing = deduped.get(key)
                if existing:
                    merged = set(existing.get("symbols", [])) | set(symbols)
                    existing["symbols"] = list(merged)
                else:
                    deduped[key] = ev
                    created_key = True
                    if not added:
                        unique.append(ev)
                        added = True
            if not created_key and not added:
                # Event fully merged into existing entries; no need to add duplicate
                continue
        return unique


__all__ = ["NewsAggregator"]
