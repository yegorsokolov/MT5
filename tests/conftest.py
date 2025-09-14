import pytest
import unittest.mock as mock
import sys
import types
import contextlib
import importlib.machinery
from pathlib import Path

# Pre-stub modules so early imports succeed
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.safe_dump = lambda *a, **k: ""
yaml_mod.__spec__ = importlib.machinery.ModuleSpec("yaml", loader=None)
sys.modules.setdefault("yaml", yaml_mod)

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

import joblib as _joblib_real
sys.modules.setdefault("joblib", _joblib_real)

env_mod = types.ModuleType("utils.environment")
env_mod.ensure_environment = lambda: None
env_mod.__spec__ = importlib.machinery.ModuleSpec("utils.environment", loader=None)
sys.modules.setdefault("utils.environment", env_mod)

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

