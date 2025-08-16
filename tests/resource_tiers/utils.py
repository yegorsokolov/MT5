import subprocess
import sys
from pathlib import Path
from typing import Dict
import types
import contextlib
import importlib.machinery
import importlib.util

sys.path.append(str(Path(__file__).resolve().parents[2]))
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = lambda *a, **kw: None
mlflow_stub.set_experiment = lambda *a, **kw: None
mlflow_stub.start_run = lambda *a, **kw: contextlib.nullcontext()
mlflow_stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_stub)

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

from model_registry import ModelRegistry, monitor, ResourceCapabilities  # type: ignore

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

    registry = ModelRegistry(monitor=monitor, auto_refresh=False)
    for task, model in expected.items():
        assert registry.get(task) == model, f"Expected {model} for {task}"

    subprocess.run([sys.executable, "-m", "py_compile", "train.py", "train_rl.py"], check=True)
