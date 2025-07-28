"""Minimal stub of the ray API for tests.

This provides just enough of the Ray interface used inside the unit tests so
that modules importing ``ray`` do not raise ``ImportError`` when the real
package is not installed."""

import types


def remote(func=None, **kwargs):
    """Decorator stub returning the original function."""
    if func is None:
        return lambda f: f
    return func


def init(*args, **kwargs):
    """No-op ``ray.init`` replacement."""
    pass


def shutdown(*args, **kwargs):
    """No-op ``ray.shutdown`` replacement."""
    pass


def get(obj):
    """Immediately return the provided object."""
    return obj


class _OptunaSearch:
    def __init__(self, *args, **kwargs):
        pass


class _SearchModule:
    optuna = types.SimpleNamespace(OptunaSearch=_OptunaSearch)


class _TuneModule:
    search = _SearchModule()

    @staticmethod
    def run(func, config=None, num_samples=1, search_alg=None, resources_per_trial=None):
        for _ in range(num_samples):
            func(config or {})
        return types.SimpleNamespace()

    @staticmethod
    def with_parameters(func, **kwargs):
        def wrapper(config):
            return func(config, **kwargs)
        return wrapper

    @staticmethod
    def report(**kwargs):
        pass


tune = _TuneModule()

import sys

# expose submodules so `from ray.tune.search.optuna import OptunaSearch` works
sys.modules[__name__ + ".tune"] = tune
sys.modules[__name__ + ".tune.search"] = tune.search
sys.modules[__name__ + ".tune.search.optuna"] = tune.search.optuna
