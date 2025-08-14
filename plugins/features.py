"""Example feature plugin placeholder.

min_cpus: 1
min_mem_gb: 0.1
requires_gpu: false
"""

MIN_CPUS = 1
MIN_MEM_GB = 0.1
REQUIRES_GPU = False

from . import register_feature


@register_feature
def add_dummy_feature(df):
    """Add a placeholder feature column if not present."""
    if "dummy" not in df.columns:
        df["dummy"] = 0
    return df
