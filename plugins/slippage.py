"""Slippage risk check plugin.

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
def check_slippage(order: dict) -> bool:
    """Ensure trade slippage is below the configured threshold.

    Parameters
    ----------
    order: dict
        Should contain `requested_price` and `filled_price` keys.
    """
    cfg = load_config()
    if not cfg.get("use_slippage_check", True):
        return True

    req = order.get("requested_price")
    filled = order.get("filled_price")
    if req is None or filled is None:
        return True

    max_slip = cfg.get("max_slippage", 0.0005)
    return abs(filled - req) <= max_slip
