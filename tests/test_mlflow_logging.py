"""Tests for MLflow configuration logging."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class _DummyRun:
    def __enter__(self) -> "_DummyRun":  # pragma: no cover - simple context
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - simple context
        return False


def _dummy_start_run(*args, **kwargs):  # pragma: no cover - helper for stub
    return _DummyRun()


_STUB_MLFLOW = types.SimpleNamespace(
    set_tracking_uri=lambda *args, **kwargs: None,
    set_experiment=lambda *args, **kwargs: None,
    start_run=_dummy_start_run,
    log_dict=lambda *args, **kwargs: None,
    log_param=lambda *args, **kwargs: None,
    log_params=lambda *args, **kwargs: None,
    log_metric=lambda *args, **kwargs: None,
    log_metrics=lambda *args, **kwargs: None,
    end_run=lambda *args, **kwargs: None,
    log_artifact=lambda *args, **kwargs: None,
    log_artifacts=lambda *args, **kwargs: None,
)


sys.modules.setdefault("mlflow", _STUB_MLFLOW)

from pydantic import BaseModel, ConfigDict

from utils import mlflow_run  # noqa: E402  (import after stubbing mlflow)
from analytics import mlflow_client  # noqa: E402  (import after stubbing mlflow)


class _DummyConfig(BaseModel):
    secret: str
    nested: dict[str, object]
    plain: int
    model_config = ConfigDict(extra="allow")

    def __init__(self) -> None:
        super().__init__(
            secret="actual-value",
            nested={"token": "real-token", "extra": "keep"},
            plain=42,
        )
        object.__setattr__(
            self,
            "_raw_config",
            {
                "secret": "secret://resolved/api_key",
                "nested": {"token": "secret://token"},
                "plain": 42,
            },
        )


class _DictConfig(dict):
    """Mapping that allows attaching raw config metadata."""


def _patch_mlflow(monkeypatch, module):
    monkeypatch.setattr(module.mlflow, "set_tracking_uri", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.mlflow, "start_run", _dummy_start_run)


def test_mlflow_run_masks_secrets(monkeypatch):
    _patch_mlflow(monkeypatch, sys.modules[mlflow_run.__module__])
    logged: dict[str, object] = {}

    def _capture(payload, artifact):
        logged["payload"] = payload
        logged["artifact"] = artifact

    monkeypatch.setattr(sys.modules[mlflow_run.__module__].mlflow, "log_dict", _capture)

    cfg = _DummyConfig()
    with mlflow_run("experiment", cfg):
        pass

    payload = logged["payload"]
    assert isinstance(payload, Mapping)
    assert payload["secret"] == "***"
    assert payload["nested"]["token"] == "***"
    assert payload["nested"]["extra"] == "keep"
    assert payload["plain"] == 42
    assert logged["artifact"] == "config.yaml"


def test_start_run_masks_secrets(monkeypatch):
    _patch_mlflow(monkeypatch, mlflow_client)
    captured: dict[str, object] = {}

    def _capture(payload, artifact):
        captured["payload"] = payload
        captured["artifact"] = artifact

    monkeypatch.setattr(mlflow_client.mlflow, "log_dict", _capture)

    cfg = _DictConfig(
        services={
            "db": {
                "password": "resolved-secret",
                "host": "db.local",
            }
        },
        feature_flag=True,
    )
    cfg._raw_config = {
        "services": {"db": {"password": "secret://db/password", "host": "db.local"}},
        "feature_flag": True,
    }

    mlflow_client.start_run("experiment", cfg)

    payload = captured["payload"]
    assert isinstance(payload, Mapping)
    assert payload["services"]["db"]["password"] == "***"
    assert payload["services"]["db"]["host"] == "db.local"
    assert payload["feature_flag"] is True
    assert captured["artifact"] == "config.yaml"
