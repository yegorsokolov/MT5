"""Plugin registration and loading helpers.

This module imports a number of built-in plugins for their side effects. Any
errors during import should not fail the entire application, so we log them
and continue.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
import inspect

from log_utils import setup_logging
from utils.resource_monitor import monitor

# Configure root logging so warnings are visible to the global logger.
setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class PluginSpec:
    """Metadata about a registered plugin.

    The ``plugin`` attribute stores the actual callable or object that
    implements the plugin. ``PluginSpec`` implements ``__call__`` so existing
    code that expects a callable continues to work transparently.
    """

    name: str
    plugin: Callable[..., Any]
    min_cpus: int = 0
    min_mem_gb: float = 0.0
    requires_gpu: bool = False
    tier: str = "lite"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
        return self.plugin(*args, **kwargs)


FEATURE_PLUGINS: List[PluginSpec] = []
MODEL_PLUGINS: List[PluginSpec] = []
RISK_CHECKS: List[PluginSpec] = []

TIERS = {"lite": 0, "standard": 1, "gpu": 2, "full": 2, "hpc": 3}


def _meets_requirements(spec: PluginSpec) -> bool:
    caps = monitor.capabilities
    if TIERS.get(monitor.capability_tier, 0) < TIERS.get(spec.tier, 0):
        return False
    if caps.cpus < spec.min_cpus:
        return False
    if caps.memory_gb < spec.min_mem_gb:
        return False
    if spec.requires_gpu and not caps.has_gpu:
        return False
    return True


def _build_spec(
    func: Callable[..., Any], *, name: Optional[str] = None, tier: str = "lite"
) -> PluginSpec:
    module = inspect.getmodule(func)
    min_cpus = getattr(module, "MIN_CPUS", 0)
    min_mem_gb = getattr(module, "MIN_MEM_GB", 0.0)
    requires_gpu = getattr(module, "REQUIRES_GPU", False)
    return PluginSpec(
        name=name or getattr(func, "__name__", str(func)),
        plugin=func,
        min_cpus=min_cpus,
        min_mem_gb=min_mem_gb,
        requires_gpu=requires_gpu,
        tier=tier,
    )


def _register_plugin(spec: PluginSpec, registry: List[PluginSpec]) -> None:
    if not _meets_requirements(spec):
        logger.info("Skipping %s plugin %s due to insufficient resources", spec.tier, spec.name)
        return
    for i, existing in enumerate(registry):
        if existing.name == spec.name:
            if TIERS.get(spec.tier, 0) >= TIERS.get(existing.tier, 0):
                registry[i] = spec
            return
    registry.append(spec)


def register_feature(
    func: Optional[Callable[..., Any]] = None,
    *,
    tier: str = "lite",
    name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        spec = _build_spec(f, name=name, tier=tier)
        _register_plugin(spec, FEATURE_PLUGINS)
        return f

    if func is None:
        return decorator
    return decorator(func)


def register_model(
    obj: Optional[Callable[..., Any]] = None,
    *,
    tier: str = "lite",
    name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        spec = _build_spec(f, name=name, tier=tier)
        _register_plugin(spec, MODEL_PLUGINS)
        return f

    if obj is None:
        return decorator
    return decorator(obj)


def register_risk_check(func: Callable[..., Any]) -> Callable[..., Any]:
    spec = _build_spec(func)
    if _meets_requirements(spec):
        RISK_CHECKS.append(spec)
    else:  # pragma: no cover - logging only
        logger.info("Skipping risk plugin %s due to insufficient resources", spec.name)
    return func

PLUGIN_MODULES = [
    'atr',
    'donchian',
    'keltner',
    'spread',
    'slippage',
    'regime_plugin',
    'finbert_sentiment',
    'fingpt_sentiment',
    'anomaly',
    'qlib_features',
    'tsfresh_features',
    'fred_features',
    'autoencoder_features',
    'deep_regime',
    'pair_trading',
    'rl_risk',
    'graph_features',
]


def _import_plugins(reload: bool = False) -> None:
    for _mod in PLUGIN_MODULES:
        try:
            name = f"{__name__}.{_mod}"
            if reload and name in sys.modules:
                importlib.reload(sys.modules[name])
            else:
                __import__(name)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to load plugin '%s'", _mod, exc_info=True)


def _setup_watcher() -> None:
    async def _watch() -> None:
        q = monitor.subscribe()
        current = monitor.capability_tier
        while True:
            tier = await q.get()
            if TIERS.get(tier, 0) > TIERS.get(current, 0):
                logger.info("Capability tier upgraded to %s; reloading plugins", tier)
                FEATURE_PLUGINS.clear()
                MODEL_PLUGINS.clear()
                RISK_CHECKS.clear()
                _import_plugins(reload=True)
                current = tier

    monitor.start()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    loop.create_task(_watch())


# Import built-in plugins so registration side effects occur
_import_plugins()
_setup_watcher()


__all__ = [
    "FEATURE_PLUGINS",
    "MODEL_PLUGINS",
    "RISK_CHECKS",
    "register_feature",
    "register_model",
    "register_risk_check",
    "PluginSpec",
]
