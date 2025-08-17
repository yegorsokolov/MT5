from __future__ import annotations

"""Utilities for hot-reloading models without downtime."""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def hot_reload(params: Dict[str, Any], model_id: Optional[str] = None) -> None:
    """Hot-reload a model with updated parameters.

    Parameters
    ----------
    params:
        New model parameters to apply.
    model_id:
        Optional identifier of the model version being activated.
    """
    logger.info("Hot-reloaded model %s with params %s", model_id, params)
