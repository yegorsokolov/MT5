import csv
import io
import time

import pandas as pd
import pytest


def test_log_ordering(log_utils_module):
    log_mod = log_utils_module
    log_mod.log_trade("buy", symbol="0", price=0, flush=True)
    for i in range(1, 5):
        log_mod.log_trade("buy", symbol=str(i), price=i)

    log_mod.shutdown_logging()

    with open(log_mod.TRADE_LOG) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert [r["symbol"] for r in rows] == [str(i) for i in range(5)]
    assert log_mod.TRADE_COUNT.count == 5


def test_log_rotation(log_utils_module, monkeypatch):
    from logging.handlers import RotatingFileHandler

    log_mod = log_utils_module
    handler = RotatingFileHandler(log_mod.TRADE_LOG, maxBytes=200, backupCount=1)
    monkeypatch.setattr(log_mod, "_trade_handler", handler, raising=False)
    monkeypatch.setattr(log_mod, "_log_decision_sync", lambda df, h: None, raising=False)

    for i in range(50):
        log_mod.log_trade("buy", symbol=str(i), price=i)

    log_mod.shutdown_logging()

    assert log_mod.TRADE_LOG.exists()
    rotated = log_mod.TRADE_LOG.parent / f"{log_mod.TRADE_LOG.name}.1"
    assert rotated.exists()


def test_flush_writes_immediately(log_utils_module):
    log_mod = log_utils_module
    log_mod.log_trade("buy", symbol="A", price=1, flush=True)

    with open(log_mod.TRADE_LOG) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows[0]["symbol"] == "A"
    log_mod.shutdown_logging()


def test_shutdown_logging_is_idempotent(log_utils_module):
    log_mod = log_utils_module
    log_mod.shutdown_logging()  # no worker yet
    log_mod.log_trade("buy", symbol="B", price=2)
    log_mod.shutdown_logging()
    log_mod.shutdown_logging()

    assert log_mod._worker_thread is None
    assert log_mod.LOG_QUEUE.empty()


def test_log_decision_encryption_roundtrip(log_utils_module):
    log_mod = log_utils_module
    df = pd.DataFrame([{"event": "decision", "score": 0.5}])

    log_mod.log_decision(df, flush=True)

    assert log_mod.DECISION_LOG.exists()
    for _ in range(10):
        if log_mod.DECISION_LOG.stat().st_size > 0:
            break
        time.sleep(0.05)
    assert log_mod.DECISION_LOG.stat().st_size > 0
    raw = log_mod.DECISION_LOG.read_bytes()
    plain = io.BytesIO()
    df.to_parquet(plain, engine="pyarrow")
    assert raw != plain.getvalue()

    recovered = log_mod.read_decisions()
    pd.testing.assert_frame_equal(
        recovered[["event", "score"]], df[["event", "score"]]
    )


def test_log_exceptions_records_errors(log_utils_module, caplog):
    log_mod = log_utils_module

    @log_mod.log_exceptions
    def _fail() -> None:
        raise RuntimeError("boom")

    assert log_mod.ERROR_COUNT.count == 0

    with caplog.at_level("INFO"):
        with pytest.raises(RuntimeError):
            _fail()

    assert log_mod.ERROR_COUNT.count == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("Start _fail" in msg for msg in messages)
    assert any("Error in _fail" in msg for msg in messages)
