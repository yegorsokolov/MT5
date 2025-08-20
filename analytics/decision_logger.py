from __future__ import annotations

"""Utility for recording model decision context.

This module enriches decision records with optional rationale fields and
stores them via :func:`log_utils.log_decision` in the encrypted decision log.
"""

from typing import Dict

import pandas as pd

from log_utils import log_decision


def log(
    df: pd.DataFrame,
    model_id: str | None = None,
    algorithm: str | None = None,
    position_size: float | None = None,
    contributions: Dict[str, float] | None = None,
    reason: str | None = None,
) -> None:
    """Enrich ``df`` with contextual fields and append to the decision log.

    Parameters
    ----------
    df:
        DataFrame describing the decision event.
    model_id:
        Identifier of the model producing the decision.
    algorithm:
        Selected algorithm name from the strategy router.
    position_size:
        Final position size recommended by the queue/risk manager.
    contributions:
        Top SHAP/feature contributions or rule triggers.
    reason:
        Optional natural-language explanation of the decision.
    """

    if df.empty:
        return
    out = df.copy()
    if model_id is not None and "model_id" not in out.columns:
        out["model_id"] = model_id
    if algorithm is not None:
        out["algorithm"] = algorithm
    if position_size is not None:
        out["position_size"] = position_size
    if contributions:
        out["contribs"] = [contributions] if len(out) == 1 else [contributions] * len(out)
    if reason is not None:
        out["reason"] = reason
    try:
        log_decision(out)
    except Exception:
        # Logging should not disrupt critical paths
        pass
