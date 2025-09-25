from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from mt5.config_models import AppConfig
except Exception:  # pragma: no cover - fallback for minimal environments
    class _AppConfigDict(dict):
        def model_dump(self) -> dict[str, Any]:
            return dict(self)

        def get(self, key: str, default: Any | None = None) -> Any:
            return dict(self).get(key, default)

    class AppConfig:  # type: ignore[override]
        def __init__(self, **data: Any) -> None:
            self.__dict__.update(data)

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> "AppConfig":
            return cls(**data)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

        def get(self, key: str, default: Any | None = None) -> Any:
            return getattr(self, key, default)

    config_models_stub = types.ModuleType("mt5.config_models")
config_models_stub.AppConfig = AppConfig
sys.modules.setdefault("mt5", types.ModuleType("mt5"))
sys.modules["mt5.config_models"] = config_models_stub

_df_probe = pd.DataFrame([[0]])
if not hasattr(_df_probe, "columns"):
    pytest.skip("pandas stub missing DataFrame.columns", allow_module_level=True)
from training.data_loader import StreamingTrainingFrame
from training.external_sources import augment_with_external_context


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - simple stub
        return None

    @property
    def text(self) -> str:
        return ""

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.calls.append((method, url))
        return _FakeResponse(self.payload)


def _build_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "strategy": {"symbols": ["XAUUSD"], "risk_per_trade": 0.01},
            "external_context": {
                "sources": [
                    {
                        "name": "macro",
                        "url": "https://example.com/macro",
                        "records_path": ["data"],
                        "timestamp_key": "date",
                        "value_key": "value",
                        "value_name": "macro_value",
                    }
                ]
            },
        }
    )


def test_augment_with_external_context_dataframe(tmp_path: Path) -> None:
    cfg = _build_config()
    payload = {"data": [{"date": "2024-01-01", "value": 1.5}, {"date": "2024-01-02", "value": 2.25}]}
    session = _FakeSession(payload)
    df = pd.DataFrame(
        {
            "Timestamp": ["2024-01-01", "2024-01-02"],
            "price": [1.0, 2.0],
        }
    )

    augmented, results = augment_with_external_context(df, cfg, tmp_path, session=session)
    assert "macro_value" in augmented.columns
    assert augmented["macro_value"].iloc[0] == 1.5
    assert results and results[0].status == "ok"
    assert session.calls == [("GET", "https://example.com/macro")]


def test_augment_with_external_context_streaming(tmp_path: Path) -> None:
    cfg = _build_config()
    payload = {"data": [{"date": "2024-01-01", "value": 1.0}]}
    session = _FakeSession(payload)
    frame = StreamingTrainingFrame(
        chunks=[pd.DataFrame({"Timestamp": ["2024-01-01"], "price": [1.0]})]
    )

    augmented, results = augment_with_external_context(frame, cfg, tmp_path, session=session)
    assert isinstance(augmented, StreamingTrainingFrame)
    assert augmented.metadata.get("external_context")
    assert results and results[0].status == "ok"
