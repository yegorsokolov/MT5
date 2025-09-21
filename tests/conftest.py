import pytest
import unittest.mock as mock
import sys
import types
import contextlib
import importlib.machinery
import inspect
from copy import deepcopy
from pathlib import Path


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
    _joblib_real = types.ModuleType("joblib")
    _joblib_real.dump = lambda *a, **k: None
    _joblib_real.load = lambda *a, **k: None
    _joblib_real.Parallel = lambda *a, **k: None
    _joblib_real.delayed = lambda func: func
sys.modules.setdefault("joblib", _joblib_real)

log_mod = types.ModuleType("log_utils")
log_mod.LOG_FILE = Path("/tmp/app.log")
log_mod.LOG_DIR = Path("/tmp")
log_mod.setup_logging = lambda: types.SimpleNamespace()
log_mod.log_exceptions = lambda f: f
log_mod.TRADE_COUNT = types.SimpleNamespace(inc=lambda: None)
log_mod.ERROR_COUNT = types.SimpleNamespace(inc=lambda: None)
log_mod.log_decision = lambda *a, **k: None
log_mod.__spec__ = importlib.machinery.ModuleSpec("log_utils", loader=None)
sys.modules.setdefault("log_utils", log_mod)

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

