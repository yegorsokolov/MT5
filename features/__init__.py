"""Feature computation registry with resource awareness.

This package provides a registry of feature modules that can be
assembled into a processing pipeline based on ``config.yaml``.  Each
module exposes a :func:`compute(df)` function returning a dataframe with
additional features.  The registry allows selective activation of
feature sets which simplifies unit testing and makes the feature
pipeline more modular.  The registry now also tracks basic hardware
requirements so operators can see which features are active or skipped
due to insufficient resources.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:  # config is optional during import in some tests
    from utils import load_config
except Exception:  # pragma: no cover - utils may not be available in tests
    load_config = lambda: {}

from utils.resource_monitor import monitor, ResourceCapabilities

from . import (
    price,
    news,
    cross_asset,
    orderbook,
    auto_indicator,
    volume,
    multi_timeframe,
    supertrend,
    adaptive_ma,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureSpec:
    """Feature description and resource requirements."""

    compute: Callable[["pd.DataFrame"], "pd.DataFrame"]
    min_cpus: int = 1
    min_mem_gb: float = 0.0
    requires_gpu: bool = False


# Registry of feature specifications
_REGISTRY: Dict[str, FeatureSpec] = {
    "price": FeatureSpec(price.compute, min_cpus=1, min_mem_gb=1.0),
    "news": FeatureSpec(news.compute, min_cpus=2, min_mem_gb=4.0),
    "cross_asset": FeatureSpec(cross_asset.compute, min_cpus=4, min_mem_gb=8.0),
    "orderbook": FeatureSpec(orderbook.compute, min_cpus=2, min_mem_gb=2.0),
    "auto_indicator": FeatureSpec(auto_indicator.compute, min_cpus=1, min_mem_gb=1.0),
    "volume": FeatureSpec(volume.compute, min_cpus=1, min_mem_gb=1.0),
    "multi_timeframe": FeatureSpec(multi_timeframe.compute, min_cpus=1, min_mem_gb=1.0),
    "supertrend": FeatureSpec(supertrend.compute, min_cpus=1, min_mem_gb=1.0),
    "adaptive_ma": FeatureSpec(adaptive_ma.compute, min_cpus=1, min_mem_gb=1.0),
}

# Holds latest status report
_STATUS: Dict[str, object] = {}

# Output directory for dashboard consumption
_REPORT_DIR = Path("reports/feature_status")


def _meets_requirements(spec: FeatureSpec, caps: ResourceCapabilities) -> bool:
    return (
        caps.cpus >= spec.min_cpus
        and caps.memory_gb >= spec.min_mem_gb
        and (caps.has_gpu or not spec.requires_gpu)
    )


def _update_status() -> None:
    """Recompute feature availability based on current resources."""

    try:
        cfg = load_config()
        enabled = set(cfg.get("features", list(_REGISTRY)))
    except Exception:  # pragma: no cover - config issues shouldn't fail
        enabled = set(_REGISTRY)

    caps = getattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0),
    )

    statuses: Dict[str, Dict[str, object]] = {}
    for name, spec in _REGISTRY.items():
        meets = _meets_requirements(spec, caps)
        status = (
            "active"
            if (name in enabled and meets)
            else ("skipped_insufficient_resources" if name in enabled else "disabled")
        )
        statuses[name] = {
            "status": status,
            "requirements": {
                "min_cpus": spec.min_cpus,
                "min_mem_gb": spec.min_mem_gb,
                "requires_gpu": spec.requires_gpu,
            },
        }

    skipped = [
        n
        for n, s in statuses.items()
        if s["status"] == "skipped_insufficient_resources"
    ]
    suggestion: Optional[str] = None
    if skipped:
        req_cpus = max(_REGISTRY[n].min_cpus for n in skipped)
        req_mem = max(_REGISTRY[n].min_mem_gb for n in skipped)
        need_gpu = any(_REGISTRY[n].requires_gpu for n in skipped)
        suggestion = (
            f"Upgrade to {req_cpus} vCPUs / {int(req_mem)} GB RAM"
            f"{' with GPU' if need_gpu else ''} to enable {', '.join(skipped)} features"
        )

    global _STATUS
    _STATUS = {
        "features": [{"name": n, **info} for n, info in statuses.items()],
        "suggestion": suggestion,
    }

    _write_report()


def _write_report() -> None:
    try:
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with (_REPORT_DIR / "latest.json").open("w", encoding="utf-8") as f:
            json.dump(_STATUS, f)
        logger.info("Feature availability: %s", _STATUS["features"])
        if _STATUS["suggestion"]:
            logger.info(_STATUS["suggestion"])
    except Exception:  # pragma: no cover - disk issues shouldn't fail
        logger.debug("Failed to write feature status", exc_info=True)


def report_status() -> Dict[str, object]:
    """Return the latest feature status report."""

    return _STATUS


def get_feature_pipeline() -> List[Callable[["pd.DataFrame"], "pd.DataFrame"]]:
    """Return the list of compute functions enabled in the config and resources."""

    _update_status()
    active = [f["name"] for f in _STATUS.get("features", []) if f["status"] == "active"]
    return [_REGISTRY[name].compute for name in active]


async def _watch_capabilities(queue: asyncio.Queue[str]) -> None:
    while True:
        await queue.get()
        _update_status()


# Initialise on import and subscribe to capability changes when possible
_update_status()
try:
    loop = asyncio.get_running_loop()
    loop.create_task(_watch_capabilities(monitor.subscribe()))
except Exception:  # pragma: no cover - no running loop during import
    pass


__all__ = ["get_feature_pipeline", "report_status"]
