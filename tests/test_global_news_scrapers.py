import json
from datetime import datetime, timedelta, timezone

import pytest
import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news.scrapers import forexfactory_news, cbc, cnn, reuters, yahoo
from news.aggregator import NewsAggregator

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

REUTERS_JSON = json.dumps({
    "items": [
        {"time": "2024-01-01T03:00:00Z", "headline": "Reuters Title", "url": "http://reuters.com/a", "symbols": ["MSFT"]}
    ]
})

YAHOO_JSON = json.dumps({
    "items": [
        {"publishedAt": "2024-01-01T04:00:00Z", "title": "Yahoo Title", "link": "http://yahoo.com/a", "symbols": ["GOOG"]}
    ]
})

def test_parsers():
    assert forexfactory_news.parse(FOREXFACTORY_HTML)[0]["symbols"] == ["USD"]
    assert cbc.parse(CBC_HTML)[0]["title"] == "CBC Title"
    assert cnn.parse(CNN_HTML)[0]["url"] == "http://cnn.com/a"
    assert reuters.parse(REUTERS_JSON)[0]["title"] == "Reuters Title"
    assert yahoo.parse(YAHOO_JSON)[0]["symbols"] == ["GOOG"]


def test_dedup_and_ttl(tmp_path):
    now = datetime.now(timezone.utc)

    async def s1():
        return [{"timestamp": now, "title": "Dup", "url": "http://a", "symbols": ["AAPL"]}]

    async def s2():
        return [{"timestamp": now, "title": "Dup", "url": "http://b", "symbols": ["AAPL"]}]

    async def old():
        return [{"timestamp": now - timedelta(hours=2), "title": "Old", "url": "http://old", "symbols": ["AAPL"]}]

    agg = NewsAggregator(cache_dir=tmp_path, ttl_hours=1)
    asyncio.run(agg.refresh(scrapers=[s1, s2, old]))

    res = agg.get_symbol_news("AAPL", now - timedelta(hours=1), now + timedelta(hours=1))
    assert len(res) == 1
    assert res[0]["title"] == "Dup"
