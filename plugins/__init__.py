"""Plugin registration and loading helpers.

This module imports a number of built-in plugins for their side effects. Any
errors during import should not fail the entire application, so we log them
and continue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List

from log_utils import setup_logging

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
        return self.plugin(*args, **kwargs)


FEATURE_PLUGINS: List[PluginSpec] = []
MODEL_PLUGINS: List[PluginSpec] = []
RISK_CHECKS: List[PluginSpec] = []


def register_feature(func: Callable[..., Any]) -> Callable[..., Any]:
    FEATURE_PLUGINS.append(PluginSpec(name=getattr(func, "__name__", str(func)), plugin=func))
    return func


def register_model(obj: Callable[..., Any]) -> Callable[..., Any]:
    MODEL_PLUGINS.append(PluginSpec(name=getattr(obj, "__name__", str(obj)), plugin=obj))
    return obj


def register_risk_check(func: Callable[..., Any]) -> Callable[..., Any]:
    RISK_CHECKS.append(PluginSpec(name=getattr(func, "__name__", str(func)), plugin=func))
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
