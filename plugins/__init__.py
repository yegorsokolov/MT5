"""Plugin registration and loading helpers.

This module imports a number of built-in plugins for their side effects. Any
errors during import should not fail the entire application, so we log them
and continue.
"""

from __future__ import annotations

import logging

from log_utils import setup_logging

# Configure root logging so warnings are visible to the global logger.
setup_logging()
logger = logging.getLogger(__name__)

FEATURE_PLUGINS: list = []
MODEL_PLUGINS: list = []
RISK_CHECKS: list = []


def register_feature(func):
    FEATURE_PLUGINS.append(func)
    return func


def register_model(obj):
    MODEL_PLUGINS.append(obj)
    return obj


def register_risk_check(func):
    RISK_CHECKS.append(func)
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
]
