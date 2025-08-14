"""Spread risk check plugin.

min_cpus: 1
min_mem_gb: 0.1
requires_gpu: false
"""

MIN_CPUS = 1
MIN_MEM_GB = 0.1
REQUIRES_GPU = False

from . import register_risk_check
from utils import load_config

@register_risk_check
def check_spread(tick: dict) -> bool:
    """Return False when the current spread exceeds the configured limit.

    Parameters
    ----------
    tick: dict
        Dictionary with at least `Bid` and `Ask` prices.
    """
    cfg = load_config()
    if not cfg.get("use_spread_check", True):
        return True

    bid = tick.get("Bid") if "Bid" in tick else tick.get("bid")
    ask = tick.get("Ask") if "Ask" in tick else tick.get("ask")
    if bid is None or ask is None:
        return True

    max_spread = cfg.get("max_spread", 0.0005)
    return (ask - bid) <= max_spread
