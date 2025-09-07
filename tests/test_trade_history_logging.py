from pathlib import Path
import sys
import types

import pandas as pd

sys.modules.pop("log_utils", None)
crypto_stub = types.ModuleType("crypto_utils")
crypto_stub._load_key = lambda *a, **k: b""  # no-op
crypto_stub.encrypt = lambda data, key: data
crypto_stub.decrypt = lambda data, key: data
sys.modules.setdefault("crypto_utils", crypto_stub)
metrics_stub = types.ModuleType("metrics")
metrics_stub.ERROR_COUNT = types.SimpleNamespace(inc=lambda: None)
metrics_stub.TRADE_COUNT = types.SimpleNamespace(inc=lambda: None)
sys.modules.setdefault("metrics", metrics_stub)

sys.path.append(str(Path(__file__).resolve().parents[1]))
import log_utils


def test_log_trade_history_dedup(monkeypatch, tmp_path):
    history = tmp_path / "trade_history.parquet"
    index = tmp_path / "order_ids.parquet"
    monkeypatch.setattr(log_utils, "TRADE_HISTORY", history)
    monkeypatch.setattr(log_utils, "ORDER_ID_INDEX", index)
    monkeypatch.setattr(log_utils, "_order_id_cache", None, raising=False)
    rec = {
        "order_id": 1,
        "Symbol": "EURUSD",
        "Timestamp": "2023-01-01T00:00:00",
        "side": "BUY",
    }
    log_utils.log_trade_history(rec)
    log_utils.log_trade_history(rec)
    df = pd.read_parquet(history)
    assert len(df) == 1
    assert df.iloc[0]["order_id"] == 1
    assert index.exists()
    ids = set(pd.read_parquet(index)["order_id"].tolist())
    assert ids == {1}


def test_log_trade_history_avoids_full_read(monkeypatch, tmp_path):
    history = tmp_path / "trade_history.parquet"
    index = tmp_path / "order_ids.parquet"
    monkeypatch.setattr(log_utils, "TRADE_HISTORY", history)
    monkeypatch.setattr(log_utils, "ORDER_ID_INDEX", index)
    monkeypatch.setattr(log_utils, "_order_id_cache", None, raising=False)

    calls = []
    orig_read = pd.read_parquet

    def spy(path, *args, **kwargs):
        calls.append(Path(path))
        return orig_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)

    rec = {
        "order_id": 1,
        "Symbol": "EURUSD",
        "Timestamp": "2023-01-01T00:00:00",
        "side": "BUY",
    }
    log_utils.log_trade_history(rec)
    log_utils.log_trade_history(rec)
    assert history not in calls

    monkeypatch.setattr(pd, "read_parquet", orig_read)
    df = pd.read_parquet(history)
    assert len(df) == 1
