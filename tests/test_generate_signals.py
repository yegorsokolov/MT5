"""Regression tests for ``mt5.generate_signals`` main entry point."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

log_utils_stub = types.ModuleType("mt5.log_utils")
log_utils_stub.setup_logging = lambda *a, **k: logging.getLogger(
    "test_generate_signals"
)
log_utils_stub.log_predictions = lambda *a, **k: None
log_utils_stub.log_exceptions = lambda func: func
log_utils_stub.LOG_DIR = Path.cwd() / "logs"
sys.modules.setdefault("mt5.log_utils", log_utils_stub)
sys.modules.setdefault("log_utils", log_utils_stub)

state_manager_stub = types.ModuleType("mt5.state_manager")
state_manager_stub.load_runtime_state = lambda *a, **k: None
state_manager_stub.migrate_runtime_state = lambda *a, **k: None
state_manager_stub.save_runtime_state = lambda *a, **k: None
state_manager_stub.legacy_runtime_state_exists = lambda *a, **k: False
sys.modules.setdefault("mt5.state_manager", state_manager_stub)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda *a, **k: {}
sys.modules.setdefault("utils", utils_stub)

market_hours_stub = types.ModuleType("utils.market_hours")
market_hours_stub.is_market_open = lambda: True
sys.modules.setdefault("utils.market_hours", market_hours_stub)

backtest_stub = types.ModuleType("mt5.backtest")
backtest_stub.run_rolling_backtest = lambda *a, **k: None
sys.modules.setdefault("mt5.backtest", backtest_stub)
sys.modules.setdefault("backtest", backtest_stub)

river_stub = types.ModuleType("river")
river_compose_stub = types.ModuleType("river.compose")
river_stub.compose = river_compose_stub
sys.modules.setdefault("river", river_stub)
sys.modules.setdefault("river.compose", river_compose_stub)

data_pkg_stub = types.ModuleType("data")
data_pkg_stub.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("data", data_pkg_stub)

data_history_stub = types.ModuleType("data.history")
data_history_stub.load_history_parquet = lambda *a, **k: pd.DataFrame()
data_history_stub.load_history_config = lambda *a, **k: pd.DataFrame()
sys.modules.setdefault("data.history", data_history_stub)

data_features_stub = types.ModuleType("data.features")
data_features_stub.make_features = lambda df, *a, **k: df
data_features_stub.make_sequence_arrays = lambda *a, **k: (np.zeros((0, 0)), None)
sys.modules.setdefault("data.features", data_features_stub)

train_rl_stub = types.ModuleType("mt5.train_rl")
train_rl_stub.TradingEnv = object
train_rl_stub.DiscreteTradingEnv = object
train_rl_stub.RLLibTradingEnv = object
train_rl_stub.HierarchicalTradingEnv = object
sys.modules.setdefault("mt5.train_rl", train_rl_stub)

signal_queue_stub = types.ModuleType("mt5.signal_queue")
signal_queue_stub.publish_dataframe_async = lambda *a, **k: None
signal_queue_stub.get_signal_backend = lambda *a, **k: None
sys.modules.setdefault("mt5.signal_queue", signal_queue_stub)

analysis_pkg_stub = types.ModuleType("analysis")
analysis_pkg_stub.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("analysis", analysis_pkg_stub)

concept_drift_stub = types.ModuleType("analysis.concept_drift")
concept_drift_stub.ConceptDriftMonitor = object
sys.modules.setdefault("analysis.concept_drift", concept_drift_stub)

from mt5 import generate_signals  # noqa: E402  (import after sys.path tweak)


class _DummyCache:
    """Lightweight cache stub used to bypass the real prediction cache."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401,D403 - simple stub
        self.maxsize = 0

    def get(self, key: int) -> None:  # pragma: no cover - trivial behaviour
        return None

    def set(self, key: int, value: object) -> None:  # pragma: no cover - trivial
        return None


class _DummyMonitor:
    """Concept drift monitor stub that records update invocations."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401,D403 - simple stub
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def update(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
        self.calls.append((args, kwargs))


class _DummyEnsemble:
    """Minimal ensemble wrapper returning zero-valued predictions."""

    def __init__(self, base_models):  # noqa: D401 - simple stub
        self.base_models = base_models

    def predict(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        return {"ensemble": np.zeros(len(data))}


class _DummyModel:
    """Single model stub that exposes ``predict_proba``."""

    model_type_ = "lgbm"
    regime_thresholds = None
    interval_params = None
    interval_q = None
    interval_coverage = None

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        rows = len(data)
        if rows == 0:
            return np.zeros((0, 2))
        return np.zeros((rows, 2))


@pytest.fixture
def run_generate_signals(monkeypatch):
    """Patch heavy dependencies so ``generate_signals.main`` can be exercised."""

    monkeypatch.setattr(generate_signals, "PredictionCache", _DummyCache)
    monkeypatch.setattr(
        generate_signals,
        "ConceptDriftMonitor",
        lambda *args, **kwargs: _DummyMonitor(),
    )
    monkeypatch.setattr(generate_signals, "log_predictions", lambda *a, **k: None)
    monkeypatch.setattr(generate_signals, "save_runtime_state", lambda *a, **k: None)
    monkeypatch.setattr(generate_signals, "load_runtime_state", lambda *a, **k: None)
    monkeypatch.setattr(generate_signals, "migrate_runtime_state", lambda *a, **k: None)
    monkeypatch.setattr(generate_signals, "legacy_runtime_state_exists", lambda *a, **k: False)
    monkeypatch.setattr(generate_signals, "get_signal_backend", lambda cfg: None)
    monkeypatch.setattr(generate_signals, "EnsembleModel", _DummyEnsemble)
    monkeypatch.setattr(
        generate_signals,
        "compute_regression_estimates",
        lambda models, data, features: {
            "abs_return": np.zeros(len(data)),
            "volatility": np.zeros(len(data)),
        },
    )
    monkeypatch.setattr(
        generate_signals,
        "load_models",
        lambda *args, **kwargs: ([_DummyModel()], ["ma_cross", "rsi_14"], None),
    )
    monkeypatch.setattr(generate_signals.backtest, "run_rolling_backtest", lambda cfg: None)
    monkeypatch.setattr(generate_signals, "is_market_open", lambda: True)
    monkeypatch.delenv("MT5_ACCOUNT_ID", raising=False)
    generate_signals._LOGGING_INITIALIZED = False

    trade_log_stub = types.ModuleType("data.trade_log")

    class _TradeLog:  # noqa: D401 - simple stub
        def __init__(self, *args, **kwargs):
            pass

        def get_open_positions(self):  # pragma: no cover - trivial
            return []

    trade_log_stub.TradeLog = _TradeLog
    monkeypatch.setitem(sys.modules, "data.trade_log", trade_log_stub)

    def _runner(
        df: pd.DataFrame,
        config: dict[str, object],
        *,
        history_exists: bool = True,
        macro_exists: bool = False,
    ):
        observed: list[pd.DataFrame] = []

        def _capture_make_features(data: pd.DataFrame) -> pd.DataFrame:
            observed.append(data.copy(deep=True))
            return data

        monkeypatch.setattr(generate_signals, "make_features", _capture_make_features)
        monkeypatch.setattr(generate_signals, "load_config", lambda: config)
        monkeypatch.setattr(
            generate_signals, "load_history_parquet", lambda path: df.copy()
        )
        monkeypatch.setattr(
            generate_signals, "load_history_config", lambda *a, **k: df.copy()
        )
        original_exists = generate_signals.Path.exists

        def _fake_exists(self):
            if self.name == "history.parquet":
                return history_exists
            if self.name == "macro.csv":
                return macro_exists
            return original_exists(self)

        monkeypatch.setattr(generate_signals.Path, "exists", _fake_exists, raising=False)
        monkeypatch.setattr(sys, "argv", ["generate-signals"])
        generate_signals.main()
        return observed

    return _runner


def test_main_skips_symbol_filter_when_column_missing(run_generate_signals):
    """Histories without ``Symbol`` should bypass the instrument filter."""

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]),
            "ma_cross": [1, 1],
            "rsi_14": [60.0, 62.0],
        }
    )
    config = {
        "account_id": "123",
        "symbol": "EURUSD",
        "ensemble_models": [],
        "model_versions": [],
    }

    observed = run_generate_signals(df, config)
    assert len(observed) == 1
    filtered = observed[0]
    assert len(filtered) == len(df)
    assert "Symbol" not in filtered.columns


def test_main_uses_symbols_list_when_primary_symbol_missing(run_generate_signals):
    """Providing ``symbols`` should retain rows even if ``symbol`` is unset."""

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]),
            "Symbol": ["EURUSD", "USDJPY"],
            "ma_cross": [1, 1],
            "rsi_14": [60.0, 65.0],
        }
    )
    config = {
        "account_id": "123",
        "symbols": ["EURUSD", "USDJPY"],
        "ensemble_models": [],
        "model_versions": [],
    }

    observed = run_generate_signals(df, config)
    assert len(observed) == 1
    filtered = observed[0]
    assert len(filtered) == len(df)
    assert filtered["Symbol"].tolist() == df["Symbol"].tolist()


def test_main_caches_history_under_log_dir(run_generate_signals, tmp_path, monkeypatch):
    """History parquet should be written beneath the configured cache dir."""

    monkeypatch.delenv("MT5_CACHE_DIR", raising=False)
    monkeypatch.setattr(generate_signals.log_utils, "LOG_DIR", tmp_path, raising=False)

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]),
            "Symbol": ["EURUSD", "EURUSD"],
            "ma_cross": [1, 1],
            "rsi_14": [60.0, 62.0],
        }
    )
    config = {
        "account_id": "123",
        "symbol": "EURUSD",
        "ensemble_models": [],
        "model_versions": [],
    }

    writes: list[Path] = []

    def _capture_to_parquet(self, path, *args, **kwargs):
        writes.append(Path(path))
        return None

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _capture_to_parquet)

    run_generate_signals(df, config, history_exists=False)

    assert writes, "history parquet was not written"
    history_path = writes[0]
    expected_dir = tmp_path / "cache"
    assert history_path.parent == expected_dir
    assert expected_dir.exists()


def test_main_loads_macro_from_cache_dir(run_generate_signals, tmp_path, monkeypatch):
    """Macro CSV lookups should resolve through the cache helper."""

    monkeypatch.delenv("MT5_CACHE_DIR", raising=False)
    monkeypatch.setattr(generate_signals.log_utils, "LOG_DIR", tmp_path, raising=False)

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]),
            "Symbol": ["EURUSD", "EURUSD"],
            "ma_cross": [1, 1],
            "rsi_14": [60.0, 62.0],
        }
    )
    config = {
        "account_id": "123",
        "symbol": "EURUSD",
        "ensemble_models": [],
        "model_versions": [],
    }

    macro_calls: list[Path] = []

    def _fake_read_csv(path, *args, **kwargs):
        macro_calls.append(Path(path))
        return pd.DataFrame(
            {
                "Timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"],
                "macro_signal": [1.0, 1.0],
            }
        )

    monkeypatch.setattr(pd, "read_csv", _fake_read_csv)

    run_generate_signals(df, config, macro_exists=True)

    assert macro_calls, "macro CSV was not read"
    expected_path = tmp_path / "cache" / "macro.csv"
    assert macro_calls[0] == expected_path
