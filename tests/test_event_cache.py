import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset


def test_ff_events_cache(monkeypatch):
    calls = []

    class DummyResp:
        def json(self):
            return []

    def fake_get(url, timeout=10):
        calls.append(url)
        return DummyResp()

    monkeypatch.setattr(dataset, "NEWS_SOURCES", ["http://fake"])
    monkeypatch.setattr(dataset.requests, "get", fake_get)
    dataset._get_ff_events.cache_clear()
    dataset._get_ff_events()
    dataset._get_ff_events()
    assert len(calls) == 1
    dataset._get_ff_events.cache_clear()
    dataset._get_ff_events()
    assert len(calls) == 2


def test_tradays_events_cache(monkeypatch):
    calls = []

    class DummyResp:
        text = "\n".join([
            "BEGIN:VEVENT",
            "DTSTART:20200101T000000Z",
            "IMPORTANCE:High",
            "CURRENCY:USD",
            "SUMMARY:event",
            "END:VEVENT",
        ])

    def fake_get(url, timeout=10):
        calls.append(url)
        return DummyResp()

    monkeypatch.setattr(dataset.requests, "get", fake_get)
    dataset._get_tradays_events.cache_clear()
    dataset._get_tradays_events()
    dataset._get_tradays_events()
    assert len(calls) == 1
    dataset._get_tradays_events.cache_clear()
    dataset._get_tradays_events()
    assert len(calls) == 2


def test_mql5_events_cache(monkeypatch):
    import types
    import sys

    calls = {"count": 0}

    def cal_hist(from_date, to_date):
        calls["count"] += 1
        return []

    dummy_mt5 = types.SimpleNamespace(
        initialize=lambda: True,
        shutdown=lambda: None,
        calendar_value_history=cal_hist,
    )

    sys.modules["MetaTrader5"] = dummy_mt5
    dataset._get_mql5_events.cache_clear()
    dataset._get_mql5_events()
    dataset._get_mql5_events()
    assert calls["count"] == 1
    dataset._get_mql5_events.cache_clear()
    dataset._get_mql5_events()
    assert calls["count"] == 2
    sys.modules.pop("MetaTrader5", None)
