import builtins
import csv
import importlib.util
import types
from pathlib import Path

import pytest
from prometheus_client import Counter, Gauge

import metrics
import log_utils


def setup_tmp_logs(tmp_path, monkeypatch):
    if isinstance(log_utils, types.SimpleNamespace):
        spec = importlib.util.spec_from_file_location(
            "log_utils_real", Path(__file__).resolve().parents[1] / "log_utils.py"
        )
        log_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(log_mod)
    else:
        log_mod = log_utils
    monkeypatch.setattr(log_mod, "LOG_DIR", tmp_path, raising=False)
    monkeypatch.setattr(log_mod, "LOG_FILE", tmp_path / "app.log", raising=False)
    monkeypatch.setattr(log_mod, "TRADE_LOG", tmp_path / "trades.csv", raising=False)
    from prometheus_client import CollectorRegistry

    registry = CollectorRegistry()
    tc = Counter("tc_test", "trade", registry=registry)
    ec = Counter("ec_test", "error", registry=registry)
    monkeypatch.setattr(log_mod, "TRADE_COUNT", tc, raising=False)
    monkeypatch.setattr(log_mod, "ERROR_COUNT", ec, raising=False)
    monkeypatch.setattr(metrics, "TRADE_COUNT", tc, raising=False)
    monkeypatch.setattr(metrics, "ERROR_COUNT", ec, raising=False)
    return log_mod, tc, ec


def test_setup_logging_patches_print(monkeypatch, tmp_path):
    log_mod, _, _ = setup_tmp_logs(tmp_path, monkeypatch)
    orig_print = builtins.print
    if hasattr(builtins, "print_orig"):
        builtins.print = builtins.print_orig
        del builtins.print_orig
    logger = log_mod.setup_logging()
    try:
        builtins.print("test")
    finally:
        builtins.print = orig_print
        if hasattr(builtins, "print_orig"):
            del builtins.print_orig


def test_log_trade_and_exception(monkeypatch, tmp_path):
    log_mod, tc, ec = setup_tmp_logs(tmp_path, monkeypatch)

    @log_mod.log_exceptions
    def fail():
        raise ValueError("x")

    with pytest.raises(ValueError):
        fail()
    assert ec._value.get() == 1

    log_mod.log_trade("buy", symbol="XAUUSD", price=1.2)
    assert tc._value.get() == 1
    path = log_mod.TRADE_LOG
    assert path.exists()
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["event"] == "buy"
    assert rows[0]["symbol"] == "XAUUSD"


