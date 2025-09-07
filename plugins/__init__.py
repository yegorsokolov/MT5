"""Lazy plugin registration and loading utilities.

Plugins are represented by :class:`PluginSpec` instances which contain
resource requirements and a :py:meth:`load` method.  Importing the
``plugins`` package only registers these specs; plugin modules are only
imported when their ``load`` method is invoked.  This avoids importing
heavy optional dependencies on machines that cannot support them.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List
import functools

from analytics.metrics_store import record_metric
# Import plugin security module directly to avoid executing ``utils.__init__``
try:
    _ps_spec = importlib.util.spec_from_file_location(
        "plugin_security", Path(__file__).resolve().parents[0].parent / "utils" / "plugin_security.py"
    )
    _ps = importlib.util.module_from_spec(_ps_spec)
    assert _ps_spec and _ps_spec.loader
    _ps_spec.loader.exec_module(_ps)  # type: ignore
    verify_plugin = _ps.verify_plugin
except Exception:  # pragma: no cover - missing optional dependencies
    def verify_plugin(*args, **kwargs):
        return None

try:  # pragma: no cover - optional during tests
    from log_utils import setup_logging
except Exception:  # pragma: no cover - fallback if heavy deps missing
    def setup_logging() -> logging.Logger:  # type: ignore
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger()

# Import ``resource_monitor`` without importing the entire ``utils`` package
# which pulls in many optional dependencies.  This mirrors the approach used in
# ``model_registry`` and keeps tests lightweight.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "resource_monitor", Path(__file__).resolve().parents[0].parent / "utils" / "resource_monitor.py"
)
_rm = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_rm)  # type: ignore
monitor = _rm.monitor

# Import plugin runner without importing full utils package
_pr_spec = importlib.util.spec_from_file_location(
    "plugin_runner", Path(__file__).resolve().parents[0].parent / "utils" / "plugin_runner.py"
)
_pr = importlib.util.module_from_spec(_pr_spec)
assert _pr_spec and _pr_spec.loader
_pr_spec.loader.exec_module(_pr)  # type: ignore
run_sandboxed = _pr.run_plugin
PluginTimeoutError = _pr.PluginTimeoutError

DEFAULT_TIMEOUT = 5.0
DEFAULT_MEM_MB: float | None = None

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses and registries
# ---------------------------------------------------------------------------

@dataclass
class PluginSpec:
    """Description of a plugin module.

    The ``loader`` callback imports the plugin module.  ``load`` first checks
    the current system capabilities exposed by :mod:`utils.resource_monitor`.
    If requirements are not met the plugin is skipped and a message is logged.
    """

    name: str
    loader: Callable[[], Any]
    min_cpus: int = 0
    min_mem_gb: float = 0.0
    requires_gpu: bool = False
    tier: str = "lite"
    _loaded: bool = False
    last_access: float = 0.0

    def _meets_requirements(self) -> bool:
        caps = monitor.capabilities
        if caps.cpus < self.min_cpus:
            self._skip_reason = f"requires >= {self.min_cpus} CPUs"
            return False
        if caps.memory_gb < self.min_mem_gb:
            self._skip_reason = f"requires >= {self.min_mem_gb}GB RAM"
            return False
        if self.requires_gpu and not caps.has_gpu:
            self._skip_reason = "requires GPU"
            return False
        return True

    def load(self) -> Any | None:
        """Import the plugin module if resources permit.

        Returns the imported module or ``None`` if the requirements are not
        satisfied.
        """

        if self._loaded:
            self.last_access = time.time()
            return self.loader()
        if not self._meets_requirements():
            logger.info("Skipping plugin %s: %s", self.name, self._skip_reason)
            return None
        module = self.loader()
        self._loaded = True
        self.last_access = time.time()
        return module


# Actual feature/model/risk callables registered by plugin modules once
# loaded.  These lists remain empty until their corresponding modules are
# imported via :py:meth:`PluginSpec.load`.
FEATURE_PLUGINS: List[Callable[..., Any]] = []
MODEL_PLUGINS: List[Callable[..., Any]] = []
RISK_CHECKS: List[Callable[..., Any]] = []


# ---------------------------------------------------------------------------
# Registration helpers used by plugin modules once they are loaded.
# ---------------------------------------------------------------------------


def register_feature(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a feature plugin executed inside a sandboxed subprocess."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        timeout = kwargs.pop("_timeout", DEFAULT_TIMEOUT)
        mem = kwargs.pop("_mem_limit_mb", DEFAULT_MEM_MB)
        return run_sandboxed(func, *args, timeout=timeout, memory_limit_mb=mem, **kwargs)

    FEATURE_PLUGINS.append(wrapper)
    return wrapper


def register_model(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a model plugin executed in a subprocess sandbox."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        timeout = kwargs.pop("_timeout", DEFAULT_TIMEOUT)
        mem = kwargs.pop("_mem_limit_mb", DEFAULT_MEM_MB)
        return run_sandboxed(func, *args, timeout=timeout, memory_limit_mb=mem, **kwargs)

    MODEL_PLUGINS.append(wrapper)
    return wrapper


def register_risk_check(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a risk check plugin executed in a subprocess sandbox."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        timeout = kwargs.pop("_timeout", DEFAULT_TIMEOUT)
        mem = kwargs.pop("_mem_limit_mb", DEFAULT_MEM_MB)
        return run_sandboxed(func, *args, timeout=timeout, memory_limit_mb=mem, **kwargs)

    RISK_CHECKS.append(wrapper)
    return wrapper


# ---------------------------------------------------------------------------
# Build ``PluginSpec`` objects for all bundled plugins.  Requirements are
# extracted by statically parsing the plugin source files so modules are not
# imported prematurely.
# ---------------------------------------------------------------------------

PLUGIN_MODULES = [
    "atr",
    "donchian",
    "keltner",
    "spread",
    "slippage",
    "regime_plugin",
    "finbert_sentiment",
    "fingpt_sentiment",
    "multilang_sentiment",
    "anomaly",
    "qlib_features",
    "tsfresh_features",
    "fred_features",
    "autoencoder_features",
    "deep_regime",
    "pair_trading",
    "rl_risk",
    "graph_features",
    "gpu_feature",  # test helper requiring a GPU
]


_req_re = {
    "cpus": re.compile(r"MIN_CPUS\s*=\s*([0-9]+)", re.MULTILINE),
    "mem": re.compile(r"MIN_MEM_GB\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.MULTILINE),
    "gpu": re.compile(r"REQUIRES_GPU\s*=\s*(True|False)", re.MULTILINE),
}


def _extract_requirements(mod: str) -> tuple[int, float, bool]:
    path = Path(__file__).with_name(f"{mod}.py")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover - defensive
        return 0, 0.0, False
    cpus = int(_req_re["cpus"].search(text).group(1)) if _req_re["cpus"].search(text) else 0
    mem = float(_req_re["mem"].search(text).group(1)) if _req_re["mem"].search(text) else 0.0
    gpu = _req_re["gpu"].search(text)
    requires_gpu = gpu.group(1) == "True" if gpu else False
    return cpus, mem, requires_gpu


PLUGIN_SPECS: List[PluginSpec] = []
for _mod in PLUGIN_MODULES:
    mod_path = Path(__file__).with_name(f"{_mod}.py")
    try:
        verify_plugin(mod_path)
    except Exception as exc:
        logger.error("Skipping plugin %s due to signature verification failure: %s", _mod, exc)
        continue

    min_cpus, min_mem, req_gpu = _extract_requirements(_mod)

    def _loader(mod=_mod):
        return importlib.import_module(f"{__name__}.{mod}")

    PLUGIN_SPECS.append(
        PluginSpec(
            name=_mod,
            loader=_loader,
            min_cpus=min_cpus,
            min_mem_gb=min_mem,
            requires_gpu=req_gpu,
        )
    )


def purge_unused_plugins(ttl: float) -> None:
    """Unload plugins not accessed within ``ttl`` seconds."""

    if ttl <= 0:
        return
    now = time.time()
    for spec in PLUGIN_SPECS:
        if spec._loaded and now - spec.last_access > ttl:
            mod_name = f"{__name__}.{spec.name}"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            spec._loaded = False
            spec.last_access = 0.0
            try:
                record_metric("plugin_unloads", 1.0)
            except Exception:
                pass


__all__ = [
    "FEATURE_PLUGINS",
    "MODEL_PLUGINS",
    "RISK_CHECKS",
    "PLUGIN_SPECS",
    "register_feature",
    "register_model",
    "register_risk_check",
    "PluginSpec",
    "purge_unused_plugins",
]
