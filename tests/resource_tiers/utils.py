import subprocess
import sys
from pathlib import Path
from typing import Dict
import types
import contextlib
import importlib.machinery
import importlib.util

sys.path.append(str(Path(__file__).resolve().parents[2]))
REPO_ROOT = Path(__file__).resolve().parents[2]
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = lambda *a, **kw: None
mlflow_stub.set_experiment = lambda *a, **kw: None
mlflow_stub.start_run = lambda *a, **kw: contextlib.nullcontext()
mlflow_stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_stub)

telemetry_stub = types.ModuleType("telemetry")
class _Tracer:
    def start_as_current_span(self, name: str):
        return contextlib.nullcontext()


telemetry_stub.get_tracer = lambda *a, **k: _Tracer()
telemetry_stub.get_meter = lambda *a, **k: types.SimpleNamespace(create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None))
sys.modules.setdefault("telemetry", telemetry_stub)

# Load resource_monitor without triggering utils/__init__
root = Path(__file__).resolve().parents[2]
rm_spec = importlib.util.spec_from_file_location("resource_monitor", root / "utils" / "resource_monitor.py")
rm = importlib.util.module_from_spec(rm_spec)
assert rm_spec and rm_spec.loader
rm_spec.loader.exec_module(rm)  # type: ignore
sys.modules.setdefault("utils.resource_monitor", rm)
utils_pkg = types.ModuleType("utils")
utils_pkg.resource_monitor = rm
utils_pkg.load_config = lambda: {}
sys.modules.setdefault("utils", utils_pkg)

# Load plugins module dynamically
plugins_spec = importlib.util.spec_from_file_location("plugins", root / "plugins" / "__init__.py")
plugins = importlib.util.module_from_spec(plugins_spec)
sys.modules.setdefault("plugins", plugins)
assert plugins_spec and plugins_spec.loader
plugins_spec.loader.exec_module(plugins)  # type: ignore


def _cpu_loader():
    mod = types.ModuleType("cpu_feature")

    @plugins.register_feature
    def cpu_feature(df):
        df = dict(df)
        df["cpu_feature"] = True
        return df

    mod.cpu_feature = cpu_feature
    return mod


def _gpu_loader():
    mod = types.ModuleType("gpu_feature")

    @plugins.register_feature
    def gpu_feature(df):
        df = dict(df)
        df["gpu_feature"] = True
        return df

    mod.gpu_feature = gpu_feature
    return mod


plugins.PLUGIN_SPECS = [
    plugins.PluginSpec(name="cpu_feature", loader=_cpu_loader, min_cpus=1, min_mem_gb=0.1, requires_gpu=False),
    plugins.PluginSpec(name="gpu_feature", loader=_gpu_loader, min_cpus=1, min_mem_gb=0.1, requires_gpu=True),
]


def _import_plugins(reload: bool = False) -> None:
    for spec in plugins.PLUGIN_SPECS:
        if reload:
            spec._loaded = False
        spec.load()


plugins._import_plugins = _import_plugins
from mt5.model_registry import ModelRegistry, monitor, ResourceCapabilities  # type: ignore

__all__ = ["run_smoke", "ResourceCapabilities"]


def run_smoke(caps: ResourceCapabilities, tier: str, expected: Dict[str, str]) -> None:
    """Run smoke tests for a given capability tier.

    Parameters
    ----------
    caps:
        Mocked system capabilities.
    tier:
        Capability tier name to apply to the global monitor.
    expected:
        Mapping of task name to the model variant expected for the tier.
    """
    monitor.capabilities = caps
    monitor.capability_tier = tier

    plugins.FEATURE_PLUGINS.clear()
    plugins.MODEL_PLUGINS.clear()
    plugins.RISK_CHECKS.clear()
    plugins._import_plugins(reload=True)

    assert plugins.FEATURE_PLUGINS, "No feature plugins loaded"

    data: Dict[str, object] = {}
    for feat in plugins.FEATURE_PLUGINS:
        data = feat(data)
    assert data, "Feature generation failed"

    registry = ModelRegistry(monitor=monitor, auto_refresh=False)
    for task, model in expected.items():
        assert registry.get(task) == model, f"Expected {model} for {task}"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(REPO_ROOT / "mt5" / "train.py"),
            str(REPO_ROOT / "mt5" / "train_rl.py"),
        ],
        check=True,
    )
