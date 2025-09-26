"""Economic calendar event utilities."""

from __future__ import annotations

import datetime as dt
import functools
import logging
import os
import time
from typing import Dict, List

import requests
from dateutil import parser as date_parser

try:
    from utils.mt5_bridge import MetaTraderImportError, load_mt5_module
except Exception:  # pragma: no cover - MT5 bridge optional when only using public feeds
    MetaTraderImportError = RuntimeError  # type: ignore

    def load_mt5_module():  # type: ignore
        import MetaTrader5 as _mt5  # type: ignore

        return _mt5

logger = logging.getLogger(__name__)

NEWS_SOURCES = [
    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
]

EVENT_CACHE_TTL = int(os.getenv("EVENT_CACHE_TTL", "3600"))


def ttl_lru_cache(ttl_seconds: int):
    """Simple TTL cache decorator using :func:`functools.lru_cache`."""

    def decorator(func):
        cached_func = functools.lru_cache(maxsize=1)(func)
        last_update = 0.0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_update
            now = time.time()
            if now - last_update > ttl_seconds:
                cached_func.cache_clear()
                last_update = now
            return cached_func(*args, **kwargs)

        def cache_clear():
            nonlocal last_update
            cached_func.cache_clear()
            last_update = 0.0

        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator


@ttl_lru_cache(EVENT_CACHE_TTL)
def _get_ff_events() -> List[dict]:
    events = []
    for url in NEWS_SOURCES:
        try:
            logger.debug("Fetching events from %s", url)
            events.extend(requests.get(url, timeout=10).json())
        except Exception as e:  # pragma: no cover - network issues
            logger.warning("Failed to fetch events from %s: %s", url, e)
            continue
    logger.debug("Fetched %d events from FF", len(events))
    return events


@ttl_lru_cache(EVENT_CACHE_TTL)
def _get_tradays_events() -> List[dict]:
    url = "https://www.tradays.com/en/economic-calendar.ics"
    try:
        logger.debug("Fetching events from %s", url)
        text = requests.get(url, timeout=10).text
    except Exception as e:  # pragma: no cover - network issues
        logger.warning("Failed to fetch events from %s: %s", url, e)
        return []
    events: List[Dict[str, str]] = []
    cur: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line == "BEGIN:VEVENT":
            cur = {}
        elif line == "END:VEVENT":
            if "DTSTART" in cur:
                events.append(
                    {
                        "date": cur["DTSTART"],
                        "impact": cur.get("IMPORTANCE", "Medium"),
                        "currency": cur.get("CURRENCY", ""),
                        "event": cur.get("SUMMARY", ""),
                    }
                )
        elif ":" in line:
            key, val = line.split(":", 1)
            cur[key] = val
    logger.debug("Fetched %d events from Tradays", len(events))
    return events


@ttl_lru_cache(EVENT_CACHE_TTL)
def _get_mql5_events() -> List[dict]:
    try:
        mt5 = load_mt5_module()
    except (MetaTraderImportError, RuntimeError):
        return []
    if not mt5.initialize():
        logger.warning("Failed to initialize MetaTrader5 for events")
        return []
    now = dt.datetime.now(tz=dt.timezone.utc)
    start = now - dt.timedelta(days=1)
    end = now + dt.timedelta(days=7)
    logger.debug("Fetching events from MetaTrader5")
    values = mt5.calendar_value_history(from_date=start, to_date=end)
    mt5.shutdown()
    if values is None:
        return []
    events: List[dict] = []
    for v in values:
        try:
            event_time = dt.datetime.fromtimestamp(v.time, tz=dt.timezone.utc)
        except Exception:
            continue
        impact = getattr(v, "importance", 1)
        impact_map = {0: "Low", 1: "Medium", 2: "High"}
        events.append(
            {
                "date": event_time.isoformat(),
                "impact": impact_map.get(impact, "Medium"),
                "currency": getattr(v, "currency", ""),
                "event": getattr(v, "event", ""),
            }
        )
    logger.debug("Fetched %d events from MetaTrader5", len(events))
    return events


@functools.lru_cache
def get_events(past_events: bool = False) -> List[dict]:
    """Download economic calendar events from multiple sources."""
    logger.info("Fetching economic events")
    events: List[dict] = []
    events.extend(_get_ff_events())
    events.extend(_get_tradays_events())
    events.extend(_get_mql5_events())

    now = dt.datetime.now(tz=dt.timezone.utc)
    filtered: List[dict] = []
    for e in events:
        try:
            date = e["date"] = (
                date_parser.parse(e["date"])
                if isinstance(e["date"], str)
                else e["date"]
            )
        except Exception:
            continue
        if past_events or date >= now:
            filtered.append(e)
    logger.info("Total events returned: %d", len(filtered))
    return filtered

__all__ = ["get_events"]
