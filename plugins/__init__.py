FEATURE_PLUGINS = []
MODEL_PLUGINS = []
RISK_CHECKS = []


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
from . import atr  # noqa: F401
from . import donchian  # noqa: F401
from . import spread  # noqa: F401
from . import slippage  # noqa: F401
