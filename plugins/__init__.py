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
