import asyncio
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news import aggregator as news_aggregator
from news.aggregator import NewsAggregator
from news.scrapers import (
    cbc,
    cnn,
    forexfactory_news,
    global_feed,
    investing,
    marketwatch,
    reuters,
    yahoo,
)


FOREXFACTORY_HTML = """
<div class="news-article" data-symbols="USD">
  <span class="timestamp">2024-01-01T00:00:00Z</span>
  <a class="title" href="http://ff.com/a">FF Title</a>
</div>
"""

CBC_HTML = """
<article data-symbols="RY">
  <time datetime="2024-01-01T01:00:00Z"></time>
  <a href="http://cbc.ca/a">CBC Title</a>
</article>
"""

CNN_HTML = """
<article data-ticker="AAPL" data-time="2024-01-01T02:00:00Z">
  <a href="http://cnn.com/a">CNN Title</a>
</article>
"""

REUTERS_JSON = json.dumps(
    {
        "items": [
            {
                "time": "2024-01-01T03:00:00Z",
                "headline": "Reuters Title",
                "url": "http://reuters.com/a",
                "symbols": ["MSFT"],
            }
        ]
    }
)

YAHOO_JSON = json.dumps(
    {
        "items": [
            {
                "publishedAt": "2024-01-01T04:00:00Z",
                "title": "Yahoo Title",
                "link": "http://yahoo.com/a",
                "symbols": ["GOOG"],
            }
        ]
    }
)

GLOBAL_JSON = json.dumps(
    {
        "items": [
            {
                "published_at": "2024-01-01T05:00:00Z",
                "title": "Global Title",
                "summary": "Markets rally on upbeat outlook",
                "url": "http://global.com/a",
                "tickers": ["TSLA", "aapl"],
                "source": "GlobalWire",
            }
        ]
    }
)

MARKETWATCH_RSS = """
<rss><channel>
  <item>
    <title>MarketWatch Title (TSLA)</title>
    <link>http://marketwatch.com/a</link>
    <pubDate>Mon, 01 Jan 2024 06:00:00 GMT</pubDate>
    <description><![CDATA[Stocks surge as Tesla ($TSLA) beats forecasts.]]></description>
  </item>
</channel></rss>
"""

INVESTING_RSS = """
<rss><channel>
  <item>
    <title>Investing.com update</title>
    <link>http://investing.com/a</link>
    <pubDate>Mon, 01 Jan 2024 07:00:00 GMT</pubDate>
    <description><![CDATA[Company (MSFT) rises on strong guidance.]]></description>
  </item>
</channel></rss>
"""


def test_parsers() -> None:
    assert forexfactory_news.parse(FOREXFACTORY_HTML)[0]["symbols"] == ["USD"]
    assert cbc.parse(CBC_HTML)[0]["title"] == "CBC Title"
    assert cnn.parse(CNN_HTML)[0]["url"] == "http://cnn.com/a"
    assert reuters.parse(REUTERS_JSON)[0]["title"] == "Reuters Title"
    assert yahoo.parse(YAHOO_JSON)[0]["symbols"] == ["GOOG"]
    parsed = global_feed.parse(GLOBAL_JSON)
    assert parsed[0]["summary"] == "Markets rally on upbeat outlook"
    assert parsed[0]["symbols"] == ["TSLA", "AAPL"]
    mw = marketwatch.parse(MARKETWATCH_RSS)
    assert mw[0]["symbols"] == ["TSLA"]
    assert "Tesla" in mw[0]["summary"]
    inv = investing.parse(INVESTING_RSS)
    assert inv[0]["symbols"] == ["MSFT"]
    assert "guidance" in inv[0]["summary"].lower()


def test_dedup_and_ttl(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)

    async def s1():
        return [
            {"timestamp": now, "title": "Dup", "url": "http://a", "symbols": ["AAPL"]}
        ]

    async def s2():
        return [
            {"timestamp": now, "title": "Dup", "url": "http://b", "symbols": ["AAPL"]}
        ]

    async def old():
        return [
            {
                "timestamp": now - timedelta(hours=2),
                "title": "Old",
                "url": "http://old",
                "symbols": ["AAPL"],
            }
        ]

    agg = NewsAggregator(cache_dir=tmp_path, ttl_hours=1)
    asyncio.run(agg.refresh(scrapers=[s1, s2, old]))

    res = agg.get_symbol_news("AAPL", now - timedelta(hours=1), now + timedelta(hours=1))
    assert len(res) == 1
    assert res[0]["title"] == "Dup"
    assert "analysis" in res[0]
    assert "sentiment" in res[0]["analysis"]
    assert "impact" in res[0]["analysis"]
    assert "severity" in res[0]["analysis"]
    assert "sentiment_effect" in res[0]["analysis"]


def test_analysis_enrichment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(timezone.utc)

    def fake_score(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": df["symbol"],
                "timestamp": df["timestamp"],
                "impact": [0.25] * len(df),
                "uncertainty": [0.05] * len(df),
            }
        )

    dummy_model = types.SimpleNamespace(score=fake_score)
    monkeypatch.setitem(sys.modules, "news.impact_model", dummy_model)
    monkeypatch.setattr(news_aggregator, "_IMPACT_MODEL", None, raising=False)

    async def enriched():
        return [
            {
                "timestamp": now,
                "title": "Tech stocks rally on strong outlook",
                "summary": "Shares surge as company beats forecasts",
                "url": "http://example.com/news",
                "symbols": ["AAPL", "MSFT"],
            }
        ]

    agg = NewsAggregator(cache_dir=tmp_path, ttl_hours=2)
    asyncio.run(agg.refresh(scrapers=[enriched]))

    res = agg.get_symbol_news("AAPL", now - timedelta(minutes=5), now + timedelta(minutes=5))
    assert len(res) == 1
    event = res[0]
    assert event["analysis"]["sentiment"] > 0
    assert event["analysis"]["impact"] == pytest.approx(0.25)
    assert event["analysis"]["uncertainty"] == pytest.approx(0.05)
    assert event["analysis"]["length_score"] >= 0
    assert event["analysis"]["length_score"] <= 1
    assert event["analysis"]["effect_minutes"] > 0
    assert event["analysis"]["effect_half_life_minutes"] > 0
    assert event["impact"] == pytest.approx(0.25)
    assert event["impact_uncertainty"] == pytest.approx(0.05)
    assert event["analysis"]["summary"]
    assert event["analysis"]["keywords"]
    assert "markets" in event["analysis"]["topics"]
    assert event["analysis"].get("ml_sentiment") is not None
    assert event["analysis"]["severity"] >= 0
    assert "sentiment_effect" in event["analysis"]

