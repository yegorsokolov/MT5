pytest_plugins = ["tests.plugins.log_archive"]

import base64
import contextlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import json
import queue
import sys
import types
import unittest.mock as mock
from copy import deepcopy
from pathlib import Path

import pytest

_tests_module = types.ModuleType("tests")
_tests_module.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("tests", _tests_module)


@pytest.fixture
def dummy_yaml(monkeypatch):
    """Provide a minimal YAML shim for tests that expect dummy behaviour."""

    import yaml

    class _DummyYaml:
        def __init__(self) -> None:
            self.load_result: object = {}
            self.dump_text: str = ""
            self.load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self.dump_calls: list[tuple[object, object | None]] = []

        def safe_load(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.load_calls.append((args, kwargs))
            return deepcopy(self.load_result)

        def safe_dump(self, data, stream=None, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.dump_calls.append((data, stream))
            if stream is None:
                return self.dump_text
            stream.write(self.dump_text)
            return None

    stub = _DummyYaml()
    monkeypatch.setattr(yaml, "safe_load", stub.safe_load)
    monkeypatch.setattr(yaml, "safe_dump", stub.safe_dump)
    return stub

mlflow_mod = types.ModuleType("mlflow")
mlflow_mod.set_tracking_uri = lambda *a, **k: None
mlflow_mod.set_experiment = lambda *a, **k: None
mlflow_mod.start_run = lambda *a, **k: contextlib.nullcontext()
mlflow_mod.log_dict = lambda *a, **k: None
mlflow_mod.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_mod)

prom_mod = types.ModuleType("prometheus_client")
prom_mod.Counter = lambda *a, **k: None
prom_mod.Gauge = lambda *a, **k: None
prom_mod.generate_latest = lambda: b""
prom_mod.CONTENT_TYPE_LATEST = "text/plain"
prom_mod.__spec__ = importlib.machinery.ModuleSpec("prometheus_client", loader=None)
sys.modules.setdefault("prometheus_client", prom_mod)

try:
    import fastapi as fastapi_mod  # type: ignore
except Exception:  # pragma: no cover - fallback when fastapi unavailable

    fastapi_mod = types.ModuleType("fastapi")

    class _StubHTTPException(Exception):
        def __init__(self, status_code: int, detail: str | None = None) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _StubRouter:
        def __init__(self) -> None:
            self._shutdown_handlers: list = []

        async def shutdown(self) -> None:
            for handler in list(self._shutdown_handlers):
                result = handler()
                if inspect.isawaitable(result):
                    await result

        async def startup(self) -> None:  # pragma: no cover - unused hook
            return

    class _StubFastAPI:
        def __init__(self, *args, **kwargs) -> None:
            self.router = _StubRouter()

        def _register(self, func):  # type: ignore[no-untyped-def]
            return func

        def get(self, _path: str):  # type: ignore[no-untyped-def]
            return self._register

        def post(self, _path: str):  # type: ignore[no-untyped-def]
            return self._register

        def on_event(self, event: str):  # type: ignore[no-untyped-def]
            def decorator(func):
                if event == "shutdown":
                    self.router._shutdown_handlers.append(func)
                return func

            return decorator

    fastapi_mod.FastAPI = _StubFastAPI
    fastapi_mod.HTTPException = _StubHTTPException
    fastapi_mod.__spec__ = importlib.machinery.ModuleSpec("fastapi", loader=None)
    sys.modules.setdefault("fastapi", fastapi_mod)

pydantic_mod = types.ModuleType("pydantic")


class _StubBaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)


pydantic_mod.BaseModel = _StubBaseModel
pydantic_mod.ValidationError = type("ValidationError", (Exception,), {})
pydantic_mod.__spec__ = importlib.machinery.ModuleSpec("pydantic", loader=None)
sys.modules.setdefault("pydantic", pydantic_mod)

slimmable_mod = types.ModuleType("models.slimmable_network")


class _StubSlimmableNetwork:
    def __init__(self, width_multipliers=(0.25, 0.5, 1.0)) -> None:
        self.width_multipliers = list(width_multipliers)
        self.active_multiplier = self.width_multipliers[-1]

    def set_width(self, width: float) -> None:
        self.active_multiplier = width


def _stub_select_width_multiplier(widths):  # type: ignore[no-untyped-def]
    return sorted(widths)[-1]


slimmable_mod.SlimmableNetwork = _StubSlimmableNetwork
slimmable_mod.select_width_multiplier = _stub_select_width_multiplier
slimmable_mod.__spec__ = importlib.machinery.ModuleSpec(
    "models.slimmable_network", loader=None
)
sys.modules.setdefault("models.slimmable_network", slimmable_mod)

filelock_mod = types.ModuleType("filelock")


class _StubFileLock:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):  # pragma: no cover - trivial context manager
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    def acquire(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
        return None

    def release(self) -> None:  # pragma: no cover - trivial
        return None


filelock_mod.FileLock = _StubFileLock
filelock_mod.__spec__ = importlib.machinery.ModuleSpec("filelock", loader=None)
sys.modules.setdefault("filelock", filelock_mod)

config_models_mod = types.ModuleType("config_models")


class _StubAppConfig:
    def __init__(self, **data) -> None:
        self.__dict__.update(data)

    def model_dump(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "_raw_config"}


config_models_mod.AppConfig = _StubAppConfig
config_models_mod.ConfigError = type("ConfigError", (Exception,), {})
config_models_mod.__spec__ = importlib.machinery.ModuleSpec("config_models", loader=None)
sys.modules.setdefault("config_models", config_models_mod)

psutil_mod = types.ModuleType("psutil")


class _StubProcess:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def cpu_percent(self) -> float:
        return 0.0

    def memory_info(self):  # pragma: no cover - trivial stub
        return types.SimpleNamespace(rss=0)


psutil_mod.Process = _StubProcess
psutil_mod.cpu_count = lambda logical=True: 4
psutil_mod.virtual_memory = lambda: types.SimpleNamespace(total=8 * (1024**3))
psutil_mod.disk_io_counters = lambda: types.SimpleNamespace(read_bytes=0, write_bytes=0)
psutil_mod.__spec__ = importlib.machinery.ModuleSpec("psutil", loader=None)
sys.modules.setdefault("psutil", psutil_mod)

metrics_mod = types.ModuleType("analytics.metrics_store")
metrics_mod.record_metric = lambda *a, **k: None
metrics_mod.model_cache_hit = lambda: None
metrics_mod.model_unload = lambda: None
metrics_mod.__spec__ = importlib.machinery.ModuleSpec("analytics.metrics_store", loader=None)
sys.modules.setdefault("analytics.metrics_store", metrics_mod)

# Additional stubs for lightweight imports
scheduler_mod = types.ModuleType("scheduler")
scheduler_mod.start_scheduler = lambda *a, **k: None
scheduler_mod.__spec__ = importlib.machinery.ModuleSpec("scheduler", loader=None)
sys.modules.setdefault("scheduler", scheduler_mod)

try:
    from mt5 import crypto_utils as crypto_utils_mod  # type: ignore
except Exception:  # pragma: no cover - fallback when crypto_utils unavailable
    crypto_utils_mod = types.ModuleType("crypto_utils")
    crypto_utils_mod._load_key = lambda *a, **k: b""
    crypto_utils_mod.encrypt = lambda *a, **k: b""
    crypto_utils_mod.decrypt = lambda *a, **k: b""
    crypto_utils_mod.__spec__ = importlib.machinery.ModuleSpec("crypto_utils", loader=None)
    sys.modules.setdefault("crypto_utils", crypto_utils_mod)

tail_mod = types.ModuleType("risk.tail_hedger")

class _StubHedger:
    def __init__(self, *a, **k):
        self.hedge_ratio = 0.0

    def evaluate(self) -> None:  # pragma: no cover - trivial
        return

tail_mod.TailHedger = _StubHedger
tail_mod.__spec__ = importlib.machinery.ModuleSpec("risk.tail_hedger", loader=None)
sys.modules.setdefault("risk.tail_hedger", tail_mod)

alert_mod = types.ModuleType("utils.alerting")
alert_mod.send_alert = lambda *a, **k: None
alert_mod.__spec__ = importlib.machinery.ModuleSpec("utils.alerting", loader=None)
sys.modules.setdefault("utils.alerting", alert_mod)

class _DF(list):
    def __init__(self, data=None):
        super().__init__(data or [])

    def to_dict(self, orient="records"):
        return list(self)

try:
    import pandas as pd_mod  # type: ignore
except Exception:  # pragma: no cover - pandas may not be installed
    pd_mod = types.ModuleType("pandas")
    pd_mod.DataFrame = _DF
    pd_mod.date_range = lambda start=None, periods=0, freq=None: [0] * periods
pd_mod.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
sys.modules["pandas"] = pd_mod

if "pandas.api" not in sys.modules:
    pandas_api_mod = types.ModuleType("pandas.api")
    pandas_api_types = types.ModuleType("pandas.api.types")
    pandas_api_types.is_numeric_dtype = lambda *a, **k: True
    pandas_api_mod.types = pandas_api_types  # type: ignore[attr-defined]
    sys.modules["pandas.api"] = pandas_api_mod
    sys.modules["pandas.api.types"] = pandas_api_types

try:
    import joblib as _joblib_real
except Exception:  # pragma: no cover - joblib may not be installed
    import pickle

    _joblib_real = types.ModuleType("joblib")

    def _joblib_dump(value, filename, *args, **kwargs):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        protocol = kwargs.get("protocol", pickle.HIGHEST_PROTOCOL)
        with path.open("wb") as fh:
            pickle.dump(value, fh, protocol=protocol)
        return [str(path)]

    def _joblib_load(filename, *args, **kwargs):
        path = Path(filename)
        with path.open("rb") as fh:
            return pickle.load(fh)

    class _JoblibParallel:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple shim
            self._backend = kwargs

        def __call__(self, tasks):  # pragma: no cover - deterministic stub
            return [task() for task in tasks]

    def _joblib_delayed(func):  # pragma: no cover - deterministic stub
        def _wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return _wrapper

    _joblib_real.dump = _joblib_dump
    _joblib_real.load = _joblib_load
    _joblib_real.Parallel = _JoblibParallel
    _joblib_real.delayed = _joblib_delayed

sys.modules.setdefault("joblib", _joblib_real)

@pytest.fixture
def log_utils_module(tmp_path, monkeypatch):
    """Load the real ``log_utils`` module inside an isolated temp directory."""

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    sys.modules.pop("log_utils", None)
    spec = importlib.util.spec_from_file_location("log_utils", root / "log_utils.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    original_columns = list(module.TRADE_COLUMNS)

    monkeypatch.setattr(module, "LOG_DIR", tmp_path, raising=False)
    monkeypatch.setattr(module, "LOG_FILE", tmp_path / "app.log", raising=False)
    monkeypatch.setattr(module, "TRADE_LOG", tmp_path / "trades.csv", raising=False)
    monkeypatch.setattr(
        module, "DECISION_LOG", tmp_path / "decisions.parquet.enc", raising=False
    )
    monkeypatch.setattr(
        module, "TRADE_HISTORY", tmp_path / "trade_history.parquet", raising=False
    )
    monkeypatch.setattr(
        module,
        "ORDER_ID_INDEX",
        tmp_path / "trade_history_order_ids.parquet",
        raising=False,
    )
    monkeypatch.setattr(module, "state_sync", None, raising=False)

    module.LOG_QUEUE = queue.Queue()
    module._worker_thread = None
    module._trade_handler = None
    class _DummyHandler:
        maxBytes = 5 * 1024 * 1024
        stream = None

        def close(self) -> None:  # pragma: no cover - simple stub
            return

    module._decision_handler = _DummyHandler()
    module._order_id_cache = None

    class _Counter:
        def __init__(self) -> None:
            self.count = 0

        def inc(self) -> None:  # pragma: no cover - trivial increment
            self.count += 1

    module.TRADE_COUNT = _Counter()
    module.ERROR_COUNT = _Counter()

    requests_stub = types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(post=lambda *a, **k: None),
        head=lambda *a, **k: None,
    )
    monkeypatch.setattr(module, "requests", requests_stub, raising=False)

    sys.modules.pop("crypto_utils", None)
    crypto_utils_mod = importlib.import_module("crypto_utils")
    from mt5.crypto_utils import encrypt as _fixture_encrypt
    import base64
    import io
    import os

    def _test_log_decision_sync(df, handler):
        if module.DECISION_LOG.exists() and module.DECISION_LOG.stat().st_size == 0:
            module.DECISION_LOG.unlink()
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow")
        payload = buf.getvalue()
        module._last_parquet_bytes = payload
        key = base64.b64decode(os.environ["DECISION_AES_KEY"])
        data = _fixture_encrypt(payload, key)
        module._last_decision_bytes = data
        with open(module.DECISION_LOG, "wb") as f:
            f.write(data)
        if module.state_sync:
            module.state_sync.sync_decisions()

    monkeypatch.setattr(module, "_log_decision_sync", _test_log_decision_sync, raising=False)

    key = base64.b64encode(b"0" * 32).decode()
    monkeypatch.setenv("DECISION_AES_KEY", key)

    sys.modules["log_utils"] = module
    sys.modules["mt5.log_utils"] = module

    yield module

    module.shutdown_logging()
    module.LOG_QUEUE = queue.Queue()
    module._order_id_cache = None
    module.TRADE_COLUMNS[:] = original_columns

scipy_mod = types.ModuleType("scipy")
stats_mod = types.ModuleType("scipy.stats")
stats_mod.genpareto = type("Genpareto", (), {})()
stats_mod.__spec__ = importlib.machinery.ModuleSpec("scipy.stats", loader=None)
scipy_mod.stats = stats_mod
scipy_mod.__spec__ = importlib.machinery.ModuleSpec("scipy", loader=None)
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.stats", stats_mod)


@pytest.fixture
def monkeypatch():
    mp = pytest.MonkeyPatch()
    patchers = []

    def patch(target, *args, **kwargs):
        p = mock.patch(target, *args, **kwargs)
        patchers.append(p)
        return p.start()

    mp.patch = patch
    yield mp
    for p in reversed(patchers):
        p.stop()
    mp.undo()

