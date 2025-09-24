import asyncio
import math
import pytest
import sys
import types
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DummyGauge:
    def __init__(self, *args, **kwargs):
        pass

    def set(self, value):
        pass

    def inc(self, value=1):
        pass


DummyCounter = DummyGauge

sys.modules.setdefault(
    "prometheus_client",
    types.SimpleNamespace(Gauge=DummyGauge, Counter=DummyCounter),
)

sys.modules.setdefault("utils", types.SimpleNamespace(load_config=lambda: {}))
sys.modules.setdefault(
    "utils.resource_monitor",
    types.SimpleNamespace(
        monitor=types.SimpleNamespace(capability_tier="lite")
    ),
)
sys.modules.setdefault(
    "analysis.strategy_evaluator",
    types.SimpleNamespace(StrategyEvaluator=object),
)


class _DummyExecutionEngine:
    def __init__(self, *args, **kwargs):
        self.volumes = []

    def start_optimizer(self):
        return None

    def stop_optimizer(self):
        return None

    def record_volume(self, value):
        self.volumes.append(value)

    async def place_order(self, *args, **kwargs):
        quantity = kwargs.get("quantity", 0.0)
        mid = kwargs.get("mid", 1.0)
        return {"filled": quantity, "avg_price": mid}

    def execute(self, *args, **kwargs):
        return types.SimpleNamespace(status="filled", price=1.0)


sys.modules.setdefault(
    "execution",
    types.SimpleNamespace(ExecutionEngine=_DummyExecutionEngine),
)
sys.modules.setdefault(
    "execution.execution_optimizer",
    types.SimpleNamespace(
        ExecutionOptimizer=type(
            "EO",
            (),
            {
                "get_params": lambda self: {"limit_offset": 0.0, "slice_size": None},
                "schedule_nightly": lambda self: None,
            },
        ),
        OptimizationLoopHandle=object,
    ),
)

sk_module = types.ModuleType("sklearn")
sk_module.pipeline = types.SimpleNamespace(Pipeline=object)
sk_module.preprocessing = types.SimpleNamespace(StandardScaler=object)
sys.modules["sklearn"] = sk_module
sys.modules["sklearn.pipeline"] = sk_module.pipeline
sys.modules["sklearn.preprocessing"] = sk_module.preprocessing

data_mod = types.ModuleType("data")
data_history = types.ModuleType("data.history")
data_history.load_history_parquet = lambda *a, **k: None
data_history.load_history_config = lambda *a, **k: None
data_features = types.ModuleType("data.features")
data_features.make_features = lambda df: df
data_feature_scaler = types.ModuleType("data.feature_scaler")
data_feature_scaler.FeatureScaler = object
data_mod.history = data_history
data_mod.features = data_features
data_mod.feature_scaler = data_feature_scaler
sys.modules["data"] = data_mod
sys.modules["data.history"] = data_history
sys.modules["data.features"] = data_features
sys.modules["data.feature_scaler"] = data_feature_scaler
crypto_utils_stub = types.SimpleNamespace(
    _load_key=lambda *a, **k: None,
    encrypt=lambda *a, **k: b"",
    decrypt=lambda *a, **k: b"",
)
sys.modules.setdefault("crypto_utils", crypto_utils_stub)
sys.modules.setdefault("mt5.crypto_utils", crypto_utils_stub)
sys.modules.setdefault(
    "state_manager",
    types.SimpleNamespace(
        load_router_state=lambda *a, **k: None, save_router_state=lambda *a, **k: None
    ),
)
sys.modules.setdefault("event_store", types.SimpleNamespace())
sys.modules.setdefault(
    "event_store.event_writer", types.SimpleNamespace(record=lambda *a, **k: None)
)
sys.modules.setdefault("model_registry", types.SimpleNamespace(ModelRegistry=object))
sys.modules.setdefault(
    "execution.execution_optimizer",
    types.SimpleNamespace(
        ExecutionOptimizer=type(
            "EO",
            (),
            {
                "get_params": lambda self: {"limit_offset": 0.0, "slice_size": None},
                "schedule_nightly": lambda self: None,
            },
        )
    ),
)
sys.modules["requests"] = types.SimpleNamespace(
    Session=lambda: types.SimpleNamespace(post=lambda *a, **k: None),
    get=lambda *a, **k: None,
)
ray_stub = types.SimpleNamespace(remote=lambda f: f)
sys.modules.setdefault(
    "ray_utils",
    types.SimpleNamespace(ray=ray_stub, init=lambda **k: None, shutdown=lambda: None),
)

tuning_module = types.ModuleType("tuning")
sys.modules.setdefault("tuning", tuning_module)
sys.modules.setdefault(
    "tuning.evolutionary_search",
    types.SimpleNamespace(run_evolutionary_search=lambda *a, **k: None),
)

# Load backtest module with dummy logging to avoid import side effects
spec = importlib.util.spec_from_file_location(
    "backtest", Path(__file__).resolve().parents[1] / "mt5" / "backtest.py"
)
sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)
backtest = importlib.util.module_from_spec(spec)
backtest.setup_logging = lambda: None
sys.modules["log_utils"] = types.SimpleNamespace(
    setup_logging=lambda: None,
    log_exceptions=lambda f: f,
    LOG_DIR=Path.cwd() / "logs",
)
spec.loader.exec_module(backtest)

backtest.init_logging = lambda: logging.getLogger("test_backtest")
backtest.log_backtest_stats = lambda metrics: None
sys.modules.setdefault("mt5.backtest", backtest)

trailing_stop = backtest.trailing_stop


def test_trailing_stop_tightens():
    stop = trailing_stop(1.0, 1.05, 0.99, 0.01)
    assert stop == pytest.approx(1.04)
    stop = trailing_stop(1.0, 1.07, stop, 0.01)
    assert stop == pytest.approx(1.06)


def test_trailing_stop_does_not_loosen():
    stop = 1.04
    new_stop = trailing_stop(1.0, 1.05, stop, 0.02)
    assert new_stop == stop


def test_compute_metrics_constant_returns():
    returns = pd.Series([0.0] * 10)
    metrics = backtest.compute_metrics(returns)
    assert metrics["sharpe"] == 0.0
    assert all(np.isfinite(list(metrics.values())))


def test_compute_metrics_near_constant_positive_returns():
    returns = pd.Series([0.01] * 10)
    metrics = backtest.compute_metrics(returns)
    assert metrics["sharpe"] == 0.0
    assert all(np.isfinite(list(metrics.values())))


def test_compute_metrics_no_trades():
    returns = pd.Series(dtype=float)
    metrics = backtest.compute_metrics(returns)
    assert metrics == {
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "return": 0.0,
        "win_rate": 0.0,
    }


def test_backtest_on_df_no_trades():
    class ZeroModel:
        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    df = pd.DataFrame(
        {
            "return": [0.0, 0.0, 0.0],
            "mid": [1.0, 1.0, 1.0],
            "Bid": [1.0, 1.0, 1.0],
            "Ask": [1.0, 1.0, 1.0],
            "BidVolume": [1.0, 1.0, 1.0],
            "AskVolume": [1.0, 1.0, 1.0],
        }
    )
    metrics = backtest.backtest_on_df(df, ZeroModel(), {})
    assert metrics["trade_count"] == 0
    assert metrics["total_return"] == 0.0
    assert metrics["return"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["win_rate"] == 0.0


def test_backtest_on_df_async_paths():
    class ZeroModel:
        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    df = pd.DataFrame(
        {
            "return": [0.0, 0.0, 0.0],
            "mid": [1.0, 1.0, 1.0],
            "Bid": [1.0, 1.0, 1.0],
            "Ask": [1.0, 1.0, 1.0],
            "BidVolume": [1.0, 1.0, 1.0],
            "AskVolume": [1.0, 1.0, 1.0],
        }
    )

    async def runner():
        metrics_async = await backtest.backtest_on_df_async(df, ZeroModel(), {})
        metrics_wrapper = await backtest.backtest_on_df(df, ZeroModel(), {})
        metrics_from_coroutine = await backtest.backtest_on_df(
            df,
            ZeroModel(),
            {},
            return_coroutine=True,
        )
        return metrics_async, metrics_wrapper, metrics_from_coroutine

    metrics_async, metrics_wrapper, metrics_from_coroutine = asyncio.run(runner())

    assert metrics_async["trade_count"] == 0
    for key, value in metrics_async.items():
        wrapper_value = metrics_wrapper[key]
        coroutine_value = metrics_from_coroutine[key]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(wrapper_value)
            assert math.isnan(coroutine_value)
        else:
            assert wrapper_value == value
            assert coroutine_value == value


class StaticProbabilityModel:
    def __init__(self, positive_probs):
        self._positive_probs = list(positive_probs)

    def predict_proba(self, X):
        length = len(X)
        if length == 0:
            return np.empty((0, 2))
        positives = np.asarray(self._positive_probs, dtype=float)
        if len(positives) == 0:
            positives = np.full(length, 0.5)
        elif len(positives) < length:
            positives = np.pad(positives, (0, length - len(positives)), mode="edge")
        else:
            positives = positives[:length]
        positives = np.clip(positives, 0.0, 1.0)
        return np.column_stack([1.0 - positives, positives])


def _make_trade_dataframe():
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=2, freq="T"),
            "return": [0.01, -0.02],
            "mid": [1.0, 0.998],
            "Bid": [0.999, 0.997],
            "Ask": [1.001, 0.999],
            "BidVolume": [100.0, 100.0],
            "AskVolume": [100.0, 100.0],
        }
    )


def _expected_single_trade_return(df: pd.DataFrame) -> float:
    entry = float(df.iloc[0]["mid"])
    exit_price = float(df.iloc[1]["mid"])
    return (exit_price - entry) / entry


def test_backtest_on_df_sync_returns_metrics():
    df = _make_trade_dataframe()
    model = StaticProbabilityModel([0.9, 0.1])

    metrics = backtest.backtest_on_df(df, model, {})

    assert isinstance(metrics, dict)
    assert metrics["trade_count"] == 1
    expected_return = _expected_single_trade_return(df)
    assert metrics["total_return"] == pytest.approx(expected_return)
    assert metrics["return"] == pytest.approx(expected_return)
    assert metrics["skipped_trades"] == 0
    assert metrics["partial_fills"] == 0
    assert math.isnan(metrics["sharpe_p_value"])


@pytest.mark.asyncio
async def test_backtest_on_df_async_returns_metrics():
    df = _make_trade_dataframe()
    model = StaticProbabilityModel([0.9, 0.1])

    metrics = await backtest.backtest_on_df_async(df, model, {})

    assert isinstance(metrics, dict)
    assert metrics["trade_count"] == 1
    expected_return = _expected_single_trade_return(df)
    assert metrics["total_return"] == pytest.approx(expected_return)
    assert metrics["return"] == pytest.approx(expected_return)
    assert metrics["skipped_trades"] == 0
    assert metrics["partial_fills"] == 0


@pytest.mark.asyncio
async def test_backtest_on_df_return_coroutine_in_running_loop():
    df = _make_trade_dataframe()
    model = StaticProbabilityModel([0.9, 0.1])

    coroutine = backtest.backtest_on_df(df, model, {}, return_coroutine=True)

    assert asyncio.iscoroutine(coroutine)

    metrics = await coroutine

    assert isinstance(metrics, dict)
    assert metrics["trade_count"] == 1
    expected_return = _expected_single_trade_return(df)
    assert metrics["total_return"] == pytest.approx(expected_return)
    assert metrics["return"] == pytest.approx(expected_return)


def test_backtest_cli_eval_fn_uses_total_return(monkeypatch):
    captured = {}

    def fake_run_backtest(cfg, external_strategy=None):
        captured["cfg"] = cfg
        return {
            "total_return": 0.123,
            "return": 9.0,
            "max_drawdown": -4.5,
            "trade_count": 7,
        }

    def fake_run_search(eval_fn, space):
        captured["space"] = space
        captured["result"] = eval_fn({"alpha": 1.0})

    monkeypatch.setattr(backtest, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(
        sys.modules["tuning.evolutionary_search"], "run_evolutionary_search", fake_run_search
    )
    monkeypatch.setattr(sys, "argv", ["backtest", "--evo-search"])

    backtest.main()

    assert captured["result"] == (-0.123, -4.5, -7.0)
    assert "alpha" in captured["cfg"]


def test_run_backtest_retains_multi_symbol_data(monkeypatch):
    df = pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB"],
            "Timestamp": pd.date_range("2024-01-01", periods=2, freq="D"),
            "return": [0.01, -0.02],
        }
    )

    original_exists = backtest.Path.exists

    def fake_exists(path_self):
        if str(path_self).endswith("history.parquet"):
            return True
        return original_exists(path_self)

    monkeypatch.setattr(backtest.Path, "exists", fake_exists)
    monkeypatch.setattr(backtest, "load_history_parquet", lambda path: df.copy())
    monkeypatch.setattr(backtest, "make_features", lambda frame: frame)
    monkeypatch.setattr(backtest.joblib, "load", lambda path: object())

    captured = {}

    def fake_backtest_on_df(dataframe, model, cfg, **kwargs):
        captured["symbols"] = dataframe["Symbol"].tolist()
        captured["rows"] = len(dataframe)
        return {"return": 0.0}

    monkeypatch.setattr(backtest, "backtest_on_df", fake_backtest_on_df)

    cfg = {"symbols": ["AAA", "BBB"], "symbol": None}
    metrics = backtest.run_backtest(cfg)

    assert metrics["return"] == 0.0
    assert captured["symbols"] == ["AAA", "BBB"]
    assert captured["rows"] == 2


def test_run_backtest_handles_missing_symbol_column(monkeypatch):
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-02-01", periods=3, freq="D"),
            "return": [0.0, 0.1, -0.05],
        }
    )

    original_exists = backtest.Path.exists

    def fake_exists(path_self):
        if str(path_self).endswith("history.parquet"):
            return True
        return original_exists(path_self)

    monkeypatch.setattr(backtest.Path, "exists", fake_exists)
    monkeypatch.setattr(backtest, "load_history_parquet", lambda path: df.copy())
    monkeypatch.setattr(backtest, "make_features", lambda frame: frame)
    monkeypatch.setattr(backtest.joblib, "load", lambda path: object())

    captured = {}

    def fake_backtest_on_df(dataframe, model, cfg, **kwargs):
        captured["rows"] = len(dataframe)
        captured["columns"] = list(dataframe.columns)
        captured["returns"] = dataframe["return"].tolist()
        return {"return": 0.0}

    monkeypatch.setattr(backtest, "backtest_on_df", fake_backtest_on_df)

    cfg = {"symbol": "EURUSD"}
    metrics = backtest.run_backtest(cfg)

    assert metrics["return"] == 0.0
    assert captured["rows"] == len(df)
    assert "Symbol" not in captured["columns"]
    assert captured["returns"] == df["return"].tolist()


def _sample_history(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=2, freq="D"),
            "return": [0.01, -0.02],
            "mid": [1.0, 1.01],
            "Bid": [0.99, 1.0],
            "Ask": [1.01, 1.02],
            "BidVolume": [100.0, 110.0],
            "AskVolume": [95.0, 90.0],
        }
    )


def _stub_backtest_execution(monkeypatch):
    monkeypatch.setattr(backtest, "make_features", lambda frame: frame)
    monkeypatch.setattr(backtest.joblib, "load", lambda path: object())
    monkeypatch.setattr(
        backtest,
        "backtest_on_df",
        lambda df, model, cfg, **kwargs: {"return": 0.0},
    )


def test_run_backtest_caches_history_in_default_log_dir(tmp_path, monkeypatch):
    symbol = "AAA"
    base_df = _sample_history(symbol)
    loaded_df = base_df.copy()
    loaded_df["Symbol"] = symbol
    log_dir = tmp_path / "logs"

    monkeypatch.setattr(backtest, "LOG_DIR", log_dir)
    monkeypatch.setattr(backtest.log_utils, "LOG_DIR", log_dir, raising=False)
    monkeypatch.setattr(
        backtest,
        "load_history_config",
        lambda sym, cfg, cfg_root: base_df.copy(),
    )

    load_calls: list[Path] = []

    def fake_load_history_parquet(path):
        load_calls.append(Path(path))
        return loaded_df.copy()

    monkeypatch.setattr(backtest, "load_history_parquet", fake_load_history_parquet)

    writes: list[Path] = []

    def fake_to_parquet(self, path, *args, **kwargs):
        writes.append(Path(path))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    _stub_backtest_execution(monkeypatch)

    metrics = backtest.run_backtest({"symbol": symbol})

    expected_path = log_dir / "cache" / "history.parquet"
    assert metrics["return"] == 0.0
    assert writes == [expected_path]
    assert load_calls == [expected_path]
    assert expected_path.parent.exists()


def test_run_backtest_honours_history_cache_override(tmp_path, monkeypatch):
    symbol = "BBB"
    base_df = _sample_history(symbol)
    loaded_df = base_df.copy()
    loaded_df["Symbol"] = symbol
    override_path = tmp_path / "custom" / "history_override.parquet"

    monkeypatch.setattr(
        backtest,
        "load_history_config",
        lambda sym, cfg, cfg_root: base_df.copy(),
    )

    load_calls: list[Path] = []

    def fake_load_history_parquet(path):
        load_calls.append(Path(path))
        return loaded_df.copy()

    monkeypatch.setattr(backtest, "load_history_parquet", fake_load_history_parquet)

    writes: list[Path] = []

    def fake_to_parquet(self, path, *args, **kwargs):
        writes.append(Path(path))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    _stub_backtest_execution(monkeypatch)

    metrics = backtest.run_backtest(
        {"symbol": symbol, "history_cache_path": str(override_path)}
    )

    assert metrics["return"] == 0.0
    assert writes == [override_path]
    assert load_calls == [override_path]
    assert override_path.parent.exists()


def test_run_backtest_reuses_supplied_model(monkeypatch):
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-03-01", periods=4, freq="D"),
            "Symbol": ["EURUSD"] * 4,
            "return": [0.01, -0.02, 0.005, 0.003],
        }
    )

    original_exists = backtest.Path.exists

    def fake_exists(path_self):
        if str(path_self).endswith("history.parquet"):
            return True
        return original_exists(path_self)

    monkeypatch.setattr(backtest.Path, "exists", fake_exists)
    monkeypatch.setattr(backtest, "load_history_parquet", lambda path: df.copy())
    monkeypatch.setattr(backtest, "make_features", lambda frame: frame)

    load_calls: list[Path] = []
    fake_model = object()

    def fake_joblib_load(path):
        load_calls.append(Path(path))
        return fake_model

    monkeypatch.setattr(backtest.joblib, "load", fake_joblib_load)

    seen_models: list[object] = []

    def fake_backtest_on_df(dataframe, model, cfg, **kwargs):
        seen_models.append(model)
        return {"return": 0.0}

    monkeypatch.setattr(backtest, "backtest_on_df", fake_backtest_on_df)

    cfg = {"symbol": "EURUSD"}
    model = backtest.joblib.load(Path("dummy.joblib"))

    backtest.run_backtest(cfg, model=model)
    backtest.run_backtest(cfg, model=model)

    assert len(load_calls) == 1
    assert seen_models == [fake_model, fake_model]
