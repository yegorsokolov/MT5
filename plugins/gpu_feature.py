"""Dummy GPU-only feature plugin for tests."""

MIN_CPUS = 1
MIN_MEM_GB = 0.1
REQUIRES_GPU = True

from . import register_feature

@register_feature
def gpu_feature(df):
    """Simple feature indicating GPU is available."""
    df = df.copy()
    df["gpu"] = True
    return df
