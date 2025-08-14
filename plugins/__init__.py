"""Plugin registration and loading helpers.

This module imports a number of built-in plugins for their side effects. Any
errors during import should not fail the entire application, so we log them
and continue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
        return self.plugin(*args, **kwargs)


FEATURE_PLUGINS: List[PluginSpec] = []
MODEL_PLUGINS: List[PluginSpec] = []
RISK_CHECKS: List[PluginSpec] = []


def _meets_requirements(spec: PluginSpec) -> bool:
    caps = monitor.capabilities
    if caps.cpus < spec.min_cpus:
        return False
    if caps.memory_gb < spec.min_mem_gb:
        return False
    if spec.requires_gpu and not caps.has_gpu:
        return False
    return True


def _build_spec(func: Callable[..., Any]) -> PluginSpec:
    module = inspect.getmodule(func)
    min_cpus = getattr(module, "MIN_CPUS", 0)
    min_mem_gb = getattr(module, "MIN_MEM_GB", 0.0)
    requires_gpu = getattr(module, "REQUIRES_GPU", False)
    return PluginSpec(
        name=getattr(func, "__name__", str(func)),
        plugin=func,
        min_cpus=min_cpus,
        min_mem_gb=min_mem_gb,
        requires_gpu=requires_gpu,
    )


def register_feature(func: Callable[..., Any]) -> Callable[..., Any]:
    spec = _build_spec(func)
    if _meets_requirements(spec):
        FEATURE_PLUGINS.append(spec)
    else:  # pragma: no cover - logging only
        logger.info("Skipping feature plugin %s due to insufficient resources", spec.name)
    return func


def register_model(obj: Callable[..., Any]) -> Callable[..., Any]:
    spec = _build_spec(obj)
    if _meets_requirements(spec):
        MODEL_PLUGINS.append(spec)
    else:  # pragma: no cover - logging only
        logger.info("Skipping model plugin %s due to insufficient resources", spec.name)
    return obj


def register_risk_check(func: Callable[..., Any]) -> Callable[..., Any]:
    spec = _build_spec(func)
    if _meets_requirements(spec):
        RISK_CHECKS.append(spec)
    else:  # pragma: no cover - logging only
        logger.info("Skipping risk plugin %s due to insufficient resources", spec.name)
    return func

# Import built-in plugins so registration side effects occur
for _mod in [
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
]:
    try:
        __import__(f"{__name__}.{_mod}")
    except Exception:  # pragma: no cover - defensive
        logger.warning("Failed to load plugin '%s'", _mod, exc_info=True)


__all__ = [
    "FEATURE_PLUGINS",
    "MODEL_PLUGINS",
    "RISK_CHECKS",
    "register_feature",
    "register_model",
    "register_risk_check",
    "PluginSpec",
]
