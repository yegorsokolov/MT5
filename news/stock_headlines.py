from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import aiohttp
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from . import impact_model
from .effect_length import estimate_effect_length

try:  # pragma: no cover - optional persistent risk settings
    from mt5.state_manager import load_user_risk  # type: ignore
except Exception:  # pragma: no cover
    load_user_risk = None  # type: ignore


# Directory for persisted headlines
_CACHE_DIR = Path("data/news_cache")
_CACHE_FILE = _CACHE_DIR / "stock_headlines.json"


@dataclass
class Headline:
    symbol: str
    timestamp: datetime
    title: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "url": self.url,
        }


async def _fetch(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch ``url`` returning text or ``None`` on error."""
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Parsing helpers

def parse_finviz(text: str, symbol: str) -> List[Headline]:
    """Parse FinViz HTML snippet into :class:`Headline` objects."""
    soup = BeautifulSoup(text, "html.parser")
    table = soup.find("table", id="news-table")
    if not table:
        return []
    headlines: List[Headline] = []
    current_date: Optional[str] = None
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 2:
            continue
        dt_str = cols[0].get_text(strip=True)
        if " " in dt_str:
            current_date, time_str = dt_str.split(" ", 1)
        else:
            time_str = dt_str
        if not current_date:
            continue
        try:
            dt = datetime.strptime(f"{current_date} {time_str}", "%b-%d-%y %I:%M%p")
        except ValueError:
            continue
        dt = dt.replace(tzinfo=timezone.utc)
        link = cols[1].find("a")
        if not link:
            continue
        title = link.get_text(strip=True)
        url = link.get("href", "")
        headlines.append(Headline(symbol.upper(), dt, title, url))
    return headlines


async def fetch_finviz(session: aiohttp.ClientSession, symbol: str) -> List[Headline]:
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    text = await _fetch(session, url)
    if not text:
        return []
    return parse_finviz(text, symbol)


async def fetch_fmp(session: aiohttp.ClientSession, symbol: str) -> List[Headline]:
    """Fetch headlines from FinancialModelingPrep JSON API."""
    url = (
        "https://financialmodelingprep.com/api/v3/stock_news?tickers="
        f"{symbol}&limit=50"
    )
    text = await _fetch(session, url)
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    headlines: List[Headline] = []
    for item in data:
        try:
            ts = datetime.strptime(item.get("publishedDate"), "%Y-%m-%d %H:%M:%S")
            ts = ts.replace(tzinfo=timezone.utc)
            headlines.append(
                Headline(
                    symbol.upper(),
                    ts,
                    item.get("title", ""),
                    item.get("url", ""),
                )
            )
        except Exception:
            continue
    return headlines


# ---------------------------------------------------------------------------
# Cache handling

def _load_cache(cache_dir: Path = _CACHE_DIR) -> List[Dict[str, Any]]:
    if not (cache_dir / _CACHE_FILE.name).exists():
        return []
    with (cache_dir / _CACHE_FILE.name).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[Dict[str, Any]] = []
    for item in raw:
        try:
            item = dict(item)
            item["timestamp"] = datetime.fromisoformat(item["timestamp"])
            out.append(item)
        except Exception:
            continue
    return out


def _save_cache(headlines: Iterable[Dict[str, Any]], cache_dir: Path = _CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    serialisable: List[Dict[str, Any]] = []
    for h in headlines:
        h = dict(h)
        ts = h.get("timestamp")
        if isinstance(ts, datetime):
            h["timestamp"] = ts.isoformat()
        serialisable.append(h)
    with (_CACHE_FILE if cache_dir == _CACHE_DIR else cache_dir / _CACHE_FILE.name).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)


def _dedupe(headlines: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {}
    for h in headlines:
        key = (h.get("symbol"), h.get("title"), h.get("timestamp"))
        if key not in seen:
            seen[key] = h
    return list(seen.values())


# ---------------------------------------------------------------------------
# Scoring

_POSITIVE = {"beat", "beats", "surge", "rally", "gain", "up"}
_NEGATIVE = {"fall", "drops", "miss", "down", "loss", "warning"}


def _risk_scale() -> float:
    if load_user_risk is None:  # pragma: no cover - optional dependency
        return 1.0
    try:
        cfg = load_user_risk()
        daily = float(cfg.get("daily_drawdown", 0.0))
        total = float(cfg.get("total_drawdown", 0.0))
        blackout = float(cfg.get("news_blackout_minutes", 0.0))
    except Exception:
        return 1.0
    if not math.isfinite(daily) or daily < 0:
        daily = 0.0
    if not math.isfinite(total) or total <= 0:
        total = max(daily, 1.0)
    ratio = max(0.05, min(1.0, daily / total if total else 0.5))
    blackout_factor = 1.0 - min(max(blackout, 0.0) / 720.0, 0.5)
    return float(max(0.4, min(1.6, (0.75 + 0.5 * ratio) * blackout_factor)))


def _sentiment(title: str) -> float:
    t = title.lower()
    pos = any(w in t for w in _POSITIVE)
    neg = any(w in t for w in _NEGATIVE)
    return float(pos) - float(neg)


def _score_headlines(headlines: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(headlines)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "title",
                "url",
                "news_movement_score",
                "severity",
                "sentiment_effect",
                "length_score",
                "effect_minutes",
                "effect_half_life",
                "risk_scale",
                "importance_score",
            ]
        )
    df["sentiment"] = df["title"].apply(_sentiment).astype(float)
    df["surprise"] = 0.0
    df["historical_response"] = 0.0
    df = df.rename(columns={"title": "text"})
    text_series = df["text"].fillna("")
    risk = _risk_scale()
    df["risk_scale"] = float(risk)
    effects = [
        estimate_effect_length(
            text=text,
            magnitude=abs(sent),
            importance=0.5,
            risk_scale=risk,
        )
        for text, sent in zip(text_series, df["sentiment"])
    ]
    df["word_count"] = text_series.str.split().str.len().fillna(0).astype(float)
    df["length_score"] = np.array([eff.score for eff in effects], dtype=float)
    df["effect_minutes"] = np.array([eff.minutes for eff in effects], dtype=float)
    df["effect_half_life"] = np.array([eff.half_life for eff in effects], dtype=float)
    df["importance_score"] = np.array([eff.importance for eff in effects], dtype=float)
    df["severity"] = np.clip(
        df["sentiment"].abs() * (0.55 + 0.25 * df["length_score"] + 0.2 * df["importance_score"]) * risk,
        0.0,
        1.0,
    )
    df["sentiment_effect"] = df["sentiment"] * df["severity"]
    scored = impact_model.score(
        df[[
            "symbol",
            "timestamp",
            "surprise",
            "sentiment",
            "historical_response",
            "text",
            "severity",
            "length_score",
            "effect_minutes",
            "effect_half_life",
            "risk_scale",
            "importance_score",
            "sentiment_effect",
        ]]
    )
    out = df.merge(scored, on=["symbol", "timestamp"])
    out = out.rename(columns={"text": "title", "impact": "news_movement_score"})
    return out.drop(columns=["uncertainty"])


# ---------------------------------------------------------------------------
# Public API

async def update_headlines(
    symbols: Iterable[str],
    cache_dir: Path | None = None,
    include_fmp: bool | None = None,
) -> List[Dict[str, Any]]:
    """Fetch headlines for ``symbols`` and persist deduped, scored cache."""

    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    if include_fmp is None:
        flag = os.getenv("ENABLE_FMP_NEWS", "").lower()
        include_fmp = bool(os.getenv("FINANCIALMODELINGPREP_API_KEY")) or flag in {
            "1",
            "true",
            "yes",
        }
    async with aiohttp.ClientSession() as session:
        tasks = []
        for sym in symbols:
            tasks.append(fetch_finviz(session, sym))
            if include_fmp:
                tasks.append(fetch_fmp(session, sym))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    headlines: List[Dict[str, Any]] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        headlines.extend([h.to_dict() for h in res])
    existing = _load_cache(cache_dir)
    headlines.extend(existing)
    deduped = _dedupe(headlines)
    scored = _score_headlines(deduped)
    _save_cache(scored.to_dict("records"), cache_dir)
    return scored.to_dict("records")


def load_scores(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load cached headline movement scores."""
    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    cached = _load_cache(cache_dir)
    if not cached:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "title",
                "url",
                "news_movement_score",
                "severity",
                "sentiment_effect",
                "length_score",
                "effect_minutes",
                "effect_half_life",
                "risk_scale",
                "importance_score",
            ]
        )
    df = pd.DataFrame(cached)
    df["news_movement_score"] = df.get("news_movement_score", 0.0).astype(float)
    df["sentiment"] = df.get("sentiment", 0.0).astype(float)
    text_series = df.get("title", pd.Series([""] * len(df))).fillna("")
    word_counts = text_series.str.split().str.len().fillna(0).astype(float)
    df["word_count"] = word_counts

    risk_default = _risk_scale()
    risk_series = df.get("risk_scale")
    if isinstance(risk_series, pd.Series):
        risk_values = risk_series.astype(float).fillna(risk_default)
    elif len(df):
        base_risk = risk_series if risk_series is not None else risk_default
        try:
            base_risk = float(base_risk)
        except (TypeError, ValueError):
            base_risk = risk_default
        risk_values = pd.Series([base_risk] * len(df), index=df.index, dtype=float)
    else:
        risk_values = pd.Series([], dtype=float)
    if risk_values.empty and len(df):
        risk_values = pd.Series([risk_default] * len(df), index=df.index, dtype=float)

    importance_series = df.get("importance_score")
    if isinstance(importance_series, pd.Series):
        importance_values = importance_series.astype(float).fillna(0.5).clip(0.0, 1.0)
    elif len(df):
        base_importance = 0.5 if importance_series is None else importance_series
        try:
            base_importance = float(base_importance)
        except (TypeError, ValueError):
            base_importance = 0.5
        importance_values = pd.Series([base_importance] * len(df), index=df.index, dtype=float)
    else:
        importance_values = pd.Series([], dtype=float)
    if importance_values.empty and len(df):
        importance_values = pd.Series([0.5] * len(df), index=df.index, dtype=float)

    effects = [
        estimate_effect_length(
            text=text,
            magnitude=abs(sent),
            importance=imp,
            risk_scale=risk,
        )
        for text, sent, imp, risk in zip(
            text_series.tolist(),
            df["sentiment"].tolist(),
            importance_values.tolist(),
            risk_values.tolist(),
        )
    ]

    df["length_score"] = np.array([eff.score for eff in effects], dtype=float)
    df["effect_minutes"] = np.array([eff.minutes for eff in effects], dtype=float)
    df["effect_half_life"] = np.array([eff.half_life for eff in effects], dtype=float)
    inferred_importance = np.array([eff.importance for eff in effects], dtype=float)
    df["importance_score"] = np.maximum(
        importance_values.to_numpy(dtype=float) if len(importance_values) else np.array([], dtype=float),
        inferred_importance,
    )
    df["risk_scale"] = risk_values.to_numpy(dtype=float) if len(risk_values) else np.array([], dtype=float)

    df["severity"] = np.clip(
        df["sentiment"].abs()
        * (0.55 + 0.25 * df["length_score"] + 0.2 * df["importance_score"])
        * df["risk_scale"],
        0.0,
        1.0,
    )
    df["sentiment_effect"] = df["sentiment"] * df["severity"]
    return df


__all__ = ["update_headlines", "load_scores", "parse_finviz"]
