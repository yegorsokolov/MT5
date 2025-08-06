import builtins
import csv
import importlib.util
import types
from pathlib import Path
import sys

import pytest
from prometheus_client import Counter, Gauge
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "metrics", Path(__file__).resolve().parents[1] / "metrics.py"
)
metrics = importlib.util.module_from_spec(spec)
sys.modules["metrics"] = metrics
spec.loader.exec_module(metrics)
spec = importlib.util.spec_from_file_location(
    "log_utils", Path(__file__).resolve().parents[1] / "log_utils.py"
)
log_utils = importlib.util.module_from_spec(spec)
sys.modules["log_utils"] = log_utils
spec.loader.exec_module(log_utils)


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
    monkeypatch.setattr(log_mod, "DECISION_LOG", tmp_path / "decisions.parquet", raising=False)
    from prometheus_client import CollectorRegistry

    registry = CollectorRegistry()
    tc = Counter("tc_test", "trade", registry=registry)
    ec = Counter("ec_test", "error", registry=registry)
    monkeypatch.setattr(log_mod, "TRADE_COUNT", tc, raising=False)
    monkeypatch.setattr(log_mod, "ERROR_COUNT", ec, raising=False)
    monkeypatch.setattr(metrics, "TRADE_COUNT", tc, raising=False)
    monkeypatch.setattr(metrics, "ERROR_COUNT", ec, raising=False)
    return log_mod, tc, ec


def test_setup_logging_returns_logger(monkeypatch, tmp_path):
    log_mod, _, _ = setup_tmp_logs(tmp_path, monkeypatch)
    orig_print = builtins.print
    logger = log_mod.setup_logging()
    assert builtins.print is orig_print
    assert logger is not None


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
    dpath = log_mod.DECISION_LOG
    assert dpath.exists()
    df = pd.read_parquet(dpath)
    assert df.loc[0, "event"] == "buy"


def test_log_predictions(monkeypatch, tmp_path):
    log_mod, _, _ = setup_tmp_logs(tmp_path, monkeypatch)
    df = pd.DataFrame({
        "Timestamp": [pd.Timestamp("2024-01-01")],
        "Symbol": ["XAUUSD"],
        "prob": [0.3],
    })
    log_mod.log_predictions(df)
    dpath = log_mod.DECISION_LOG
    assert dpath.exists()
    out = pd.read_parquet(dpath)
    assert out.loc[0, "prob"] == 0.3
    assert out.loc[0, "event"] == "prediction"


