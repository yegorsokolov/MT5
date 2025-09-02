import asyncio
from pathlib import Path
import sys
import types

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

dummy = types.ModuleType("impact_model")
dummy.get_impact = lambda *args, **kwargs: (None, 0.0)
dummy.score = lambda df: df
sys.modules.setdefault("news.impact_model", dummy)

from news import stock_headlines

FINVIZ_HTML = """
<table id="news-table">
  <tr>
    <td>Jun-07-24 01:11PM</td>
    <td><a href="http://example.com/a">Apple beats expectations</a></td>
  </tr>
  <tr>
    <td>01:00PM</td>
    <td><a href="http://example.com/b">Apple shares fall on weak guidance</a></td>
  </tr>
</table>
"""


@pytest.mark.asyncio
async def test_update_and_cache(tmp_path, monkeypatch):
    async def fake_fetch(session, url):
        return FINVIZ_HTML

    async def fake_fetch_fmp(session, symbol):
        return []

    def fake_score(df):
        return pd.DataFrame({
            "symbol": df["symbol"],
            "timestamp": df["timestamp"],
            "impact": [0.5] * len(df),
            "uncertainty": [0.0] * len(df),
        })

    monkeypatch.setattr(stock_headlines, "_fetch", fake_fetch)
    monkeypatch.setattr(stock_headlines, "fetch_fmp", fake_fetch_fmp)
    monkeypatch.setattr(stock_headlines.impact_model, "score", fake_score)

    await stock_headlines.update_headlines(["AAPL"], cache_dir=tmp_path)
    df = stock_headlines.load_scores(cache_dir=tmp_path)
    assert len(df) == 2
    assert set(df["symbol"]) == {"AAPL"}
    assert all(df["news_movement_score"] == 0.5)
    titles = set(df["title"])
    assert "Apple beats expectations" in titles
    assert "Apple shares fall on weak guidance" in titles
