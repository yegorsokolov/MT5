"""Example feature plugin placeholder.

min_cpus: 1
min_mem_gb: 0.1
requires_gpu: false
"""

MIN_CPUS = 1
MIN_MEM_GB = 0.1
REQUIRES_GPU = False

from . import register_feature


@register_feature(name="add_dummy_feature", tier="full")
def add_dummy_feature_full(df):
    """Full variant writing ones to the dummy column."""
    if "dummy" not in df.columns:
        df["dummy"] = 1
    return df


@register_feature(name="add_dummy_feature", tier="lite")
def add_dummy_feature_lite(df):
    """Lite variant writing zeros to the dummy column."""
    if "dummy" not in df.columns:
        df["dummy"] = 0
    return df
