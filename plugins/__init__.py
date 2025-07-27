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
for _mod in [
    'atr',
    'donchian',
    'keltner',
    'spread',
    'slippage',
    'regime_plugin',
    'finbert_sentiment',
    'qlib_features',
]:
    try:
        __import__(f'{__name__}.{_mod}')
    except Exception:
        pass
