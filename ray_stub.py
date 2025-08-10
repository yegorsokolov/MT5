"""Minimal stub of the Ray API for tests and offline execution."""

import types


def remote(func=None, **kwargs):
    """Decorator stub returning the original function."""
    if func is None:
        return lambda f: f
    return func


def init(*args, **kwargs):
    """No-op ``ray.init`` replacement."""
    return None


def shutdown(*args, **kwargs):
    """No-op ``ray.shutdown`` replacement."""
    return None


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
        return None


tune = _TuneModule()

__all__ = ["remote", "init", "shutdown", "get", "tune"]
