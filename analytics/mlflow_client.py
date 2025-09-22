"""Wrapper around mlflow with configurable tracking server."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any
import os
from contextlib import contextmanager

import mlflow
from mt5 import log_utils
from utils import PROJECT_ROOT, sanitize_config


def _configure(experiment: str, cfg: Mapping[str, Any]) -> None:
    """Configure mlflow tracking URI and experiment.

    If ``cfg['mlflow']['tracking_uri']`` is provided, connect to that server and
    optionally set basic-auth credentials. Otherwise logs are written under the
    local ``logs/mlruns`` directory.
    """

    ml_cfg = cfg.get("mlflow", {}) if isinstance(cfg, Mapping) else {}
    uri = ml_cfg.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
        user = ml_cfg.get("username")
        pwd = ml_cfg.get("password")
        if user:
            os.environ["MLFLOW_TRACKING_USERNAME"] = str(user)
        if pwd:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = str(pwd)
    else:
        logs_dir = getattr(log_utils, "LOG_DIR", PROJECT_ROOT / "logs")
        mlflow.set_tracking_uri(f"file:{logs_dir / 'mlruns'}")
    mlflow.set_experiment(experiment)


def start_run(experiment: str, cfg: Mapping[str, Any]) -> bool:
    """Start an MLflow run and log the configuration.

    Returns ``True`` when a new run was started. When an MLflow run is already
    active (for example because an orchestrator opened it) the existing run is
    reused and ``False`` is returned.
    """

    try:  # pragma: no cover - mlflow optional
        active = getattr(mlflow, "active_run", None)
        if callable(active):
            if active() is not None:
                return False
    except Exception:  # noqa: BLE001
        pass

    _configure(experiment, cfg)
    mlflow.start_run()
    raw_cfg = getattr(cfg, "_raw_config", None)
    sanitized = sanitize_config(cfg, raw_cfg=raw_cfg)
    payload = (
        sanitized
        if isinstance(sanitized, Mapping)
        else raw_cfg
        if isinstance(raw_cfg, Mapping)
        else dict(cfg)
    )
    mlflow.log_dict(payload, "config.yaml")
    return True


def end_run() -> None:
    """End the current MLflow run."""
    mlflow.end_run()


@contextmanager
def run(experiment: str, cfg: Mapping[str, Any]):
    """Context manager that starts an MLflow run when necessary."""

    started = False
    try:
        started = start_run(experiment, cfg)
        yield started
    finally:
        if started:
            end_run()


def log_param(key: str, value: Any) -> None:
    mlflow.log_param(key, value)


def log_params(params: Mapping[str, Any]) -> None:
    mlflow.log_params(params)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    mlflow.log_metric(key, value, step=step)


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)


def log_artifacts(path: str) -> None:
    mlflow.log_artifacts(path)
