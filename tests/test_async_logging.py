import base64
import csv
import importlib.util
import os
import queue
import sys
import types
from pathlib import Path

import pytest


def setup_tmp_logs(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))
    crypto_stub = types.SimpleNamespace(
        _load_key=lambda *a, **k: b"",
        encrypt=lambda d, k: d,
        decrypt=lambda d, k: d,
    )
    sys.modules.setdefault("crypto_utils", crypto_stub)
    spec = importlib.util.spec_from_file_location("log_utils", root / "log_utils.py")
    log_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_mod)

    monkeypatch.setattr(log_mod, "LOG_DIR", tmp_path, raising=False)
    monkeypatch.setattr(log_mod, "LOG_FILE", tmp_path / "app.log", raising=False)
    monkeypatch.setattr(log_mod, "TRADE_LOG", tmp_path / "trades.csv", raising=False)
    monkeypatch.setattr(log_mod, "DECISION_LOG", tmp_path / "decisions.parquet.enc", raising=False)
    monkeypatch.setattr(log_mod, "state_sync", None, raising=False)

    os.environ["DECISION_AES_KEY"] = base64.b64encode(b"0" * 32).decode()
    log_mod.TRADE_COUNT = types.SimpleNamespace(inc=lambda: None)
    log_mod.ERROR_COUNT = types.SimpleNamespace(inc=lambda: None)

    log_mod.LOG_QUEUE = queue.Queue()
    log_mod._worker_thread = None
    log_mod._trade_handler = None
    log_mod._decision_handler = None
    monkeypatch.setattr(log_mod, "_log_decision_sync", lambda df, h: None, raising=False)
    return log_mod


def test_log_ordering(monkeypatch, tmp_path):
    log_mod = setup_tmp_logs(tmp_path, monkeypatch)
    for i in range(5):
        log_mod.log_trade("buy", symbol=str(i), price=i)
    log_mod.shutdown_logging()
    with open(log_mod.TRADE_LOG) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert [r["symbol"] for r in rows] == [str(i) for i in range(5)]


def test_log_rotation(monkeypatch, tmp_path):
    from logging.handlers import RotatingFileHandler

    log_mod = setup_tmp_logs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        log_mod,
        "_trade_handler",
        RotatingFileHandler(log_mod.TRADE_LOG, maxBytes=200, backupCount=1),
        raising=False,
    )
    for i in range(50):
        log_mod.log_trade("buy", symbol=str(i), price=i)
    log_mod.shutdown_logging()
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "trades.csv.1").exists()
