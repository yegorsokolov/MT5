import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis import replay


def _setup_trade_and_news(tmp_path: Path, sentiment: float):
    trade_dir = tmp_path / "reports"
    trade_dir.mkdir()
    trade_path = trade_dir / "trades.csv"
    trades = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "symbol": ["XYZ"],
            "pnl": [1.0],
        }
    )
    trades.to_csv(trade_path, index=False)

    news_cache = tmp_path / "data" / "news_cache"
    news_cache.mkdir(parents=True)
    headline = {
        "symbol": "XYZ",
        "timestamp": "2024-01-01T00:00:00",
        "title": "XYZ rallies",
        "url": "",
        "sentiment": sentiment,
        "news_movement_score": 0.0,
    }
    with (news_cache / "stock_headlines.json").open("w", encoding="utf-8") as f:
        json.dump([headline], f)
    return trade_path, news_cache


def test_news_replay_reproduces_original(tmp_path):
    trade_path, news_cache = _setup_trade_and_news(tmp_path, sentiment=0.0)
    out_dir = tmp_path / "reports" / "news_replay"
    summary = replay.news_replay(
        trade_path=trade_path, news_cache_dir=news_cache, out_dir=out_dir
    )
    assert summary["pnl_with_news"].iloc[0] == summary["pnl_without_news"].iloc[0]


def test_news_replay_highlights_difference(tmp_path):
    trade_path, news_cache = _setup_trade_and_news(tmp_path, sentiment=0.5)
    out_dir = tmp_path / "reports" / "news_replay"
    summary = replay.news_replay(
        trade_path=trade_path, news_cache_dir=news_cache, out_dir=out_dir
    )
    assert summary["pnl_with_news"].iloc[0] != summary["pnl_without_news"].iloc[0]
    comp = pd.read_csv(out_dir / "trade_comparison.csv")
    assert "pnl_delta" in comp.columns
    assert comp["pnl_delta"].iloc[0] != 0
