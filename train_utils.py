from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Any, Optional

from log_utils import setup_logging
from utils import load_config
from analytics import mlflow_client as mlflow


def setup_training(config: Optional[str | Path | Mapping[str, Any]] = None, *, experiment: Optional[str] = None) -> Mapping[str, Any]:
    """Configure logging, load config and start an MLflow run.

    Parameters
    ----------
    config:
        Optional configuration path or mapping.  When ``None`` the default
        ``config.yaml`` is loaded.  If a mapping is supplied it is returned as
        is.
    experiment:
        Optional MLflow experiment name.  When provided, ``mlflow.start_run`` is
        invoked and the configuration is logged.

    Returns
    -------
    Mapping[str, Any]
        The resolved configuration dictionary.
    """
    setup_logging()
    if isinstance(config, (str, Path)):
        cfg_obj = load_config(config)
        cfg: Mapping[str, Any] = cfg_obj.model_dump()  # type: ignore[attr-defined]
    elif config is None:
        cfg_obj = load_config()
        cfg = cfg_obj.model_dump()  # type: ignore[attr-defined]
    else:
        cfg = dict(config)
    if experiment:
        try:
            mlflow.start_run(experiment, cfg)
        except Exception:  # pragma: no cover - mlflow optional
            pass
    return cfg


def end_training() -> None:
    """End the current MLflow run if one is active."""
    try:
        mlflow.end_run()
    except Exception:  # pragma: no cover - mlflow optional
        pass
