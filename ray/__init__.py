"""Minimal stub of the ray API for tests."""

def remote(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func

def init(*args, **kwargs):
    pass

def shutdown(*args, **kwargs):
    pass

def get(obj):
    return obj
