"""Feature computation registry.

This package provides a registry of feature modules that can be
assembled into a processing pipeline based on ``config.yaml``.  Each
module exposes a :func:`compute(df)` function returning a dataframe with
additional features.  The registry allows selective activation of
feature sets which simplifies unit testing and makes the feature
pipeline more modular.
"""

from __future__ import annotations

from typing import Callable, List

try:  # config is optional during import in some tests
    from utils import load_config
except Exception:  # pragma: no cover - utils may not be available in tests
    load_config = lambda: {}

from . import price, news, cross_asset

# Mapping of module name to its compute function
_REGISTRY = {
    "price": price.compute,
    "news": news.compute,
    "cross_asset": cross_asset.compute,
}


def get_feature_pipeline() -> List[Callable[["pd.DataFrame"], "pd.DataFrame"]]:
    """Return the list of compute functions enabled in the config."""
    try:
        cfg = load_config()
        enabled = cfg.get("features", list(_REGISTRY))
    except Exception:  # pragma: no cover - config issues shouldn't fail
        enabled = list(_REGISTRY)
    return [func for name, func in _REGISTRY.items() if name in enabled]

__all__ = ["get_feature_pipeline"]
