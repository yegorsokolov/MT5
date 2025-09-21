import pandas as pd
import pytest


if not hasattr(pd.DataFrame, "to_parquet") or not hasattr(pd, "read_parquet"):
    pytest.skip("requires pandas parquet support", allow_module_level=True)


def _make_record(order_id: int) -> dict:
    return {
        "order_id": order_id,
        "Symbol": "EURUSD",
        "Timestamp": "2023-01-01T00:00:00",
        "side": "BUY",
    }


def test_log_trade_history_dedup(log_utils_module):
    log_mod = log_utils_module
    rec = _make_record(1)
    log_mod.log_trade_history(rec)
    log_mod.log_trade_history(rec)

    df = pd.read_parquet(log_mod.TRADE_HISTORY)
    assert len(df) == 1
    assert df.iloc[0]["order_id"] == 1

    index = pd.read_parquet(log_mod.ORDER_ID_INDEX)
    assert set(index["order_id"].tolist()) == {1}


def test_log_trade_history_avoids_full_read(log_utils_module, monkeypatch):
    log_mod = log_utils_module
    calls: list = []

    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)

    rec = _make_record(2)
    log_mod.log_trade_history(rec)
    log_mod.log_trade_history(rec)

    assert log_mod.TRADE_HISTORY not in calls


def test_order_id_cache_persists_between_invocations(log_utils_module):
    log_mod = log_utils_module
    rec = _make_record(3)

    log_mod.log_trade_history(rec)
    cache = log_mod._load_order_id_index()
    assert 3 in cache

    log_mod._order_id_cache = None
    reloaded = log_mod._load_order_id_index()
    assert 3 in reloaded

