"""Fetch and merge external context data for training."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:  # pragma: no cover - requests may be optional in some deployments
    import requests
except Exception:  # pragma: no cover - optional dependency guard
    requests = None  # type: ignore[assignment]

from mt5.config_models import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class ExternalSourceResult:
    """Structured metadata describing an external source fetch."""

    name: str
    status: str
    rows: int
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "status": self.status, "rows": self.rows}
        if self.message:
            payload["message"] = self.message
        return payload


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, "")
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve_env_placeholders(v) for v in value]
    return value


def _navigate_path(payload: Any, path: Iterable[str] | None) -> Any:
    data = payload
    if path:
        for key in path:
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return None
    return data


def _ensure_dataframe(records: Any) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, (list, tuple)):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        return pd.DataFrame(records)
    raise TypeError(f"Unsupported records payload: {type(records)!r}")


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def augment_with_external_context(
    data: pd.DataFrame | "StreamingTrainingFrame",
    cfg: AppConfig,
    root: Path,
    *,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame | "StreamingTrainingFrame", list[ExternalSourceResult]]:
    """Attach external datasets to ``data`` when configured."""

    del root  # root is reserved for future caching features

    context_cfg = getattr(cfg, "external_context", None) or getattr(
        cfg, "__pydantic_extra__", {}
    ).get("external_context")
    if not context_cfg:
        return data, []

    if isinstance(context_cfg, dict):
        model = cfg.__class__
        cfg = model.model_validate({**cfg.model_dump(), "external_context": context_cfg})
        context_cfg = cfg.external_context

    enabled = getattr(context_cfg, "enabled", True)
    sources = getattr(context_cfg, "sources", [])
    join_type = getattr(context_cfg, "join", "left")
    if not enabled or not sources:
        return data, []

    if requests is None and session is None:
        logger.warning("Skipping external context collection because requests is unavailable")
        return data, []

    sess = session or requests.Session()
    results: list[ExternalSourceResult] = []
    frames: dict[str, pd.DataFrame] = {}

    for source in sources:
        spec = source
        if not getattr(spec, "enabled", True):
            continue
        name = getattr(spec, "name", "external")
        if not getattr(spec, "url", None):
            results.append(
                ExternalSourceResult(
                    name=name, status="skipped", rows=0, message="Missing URL"
                )
            )
            continue
        url = getattr(spec, "url")
        method = getattr(spec, "method", "GET").upper()
        fmt = getattr(spec, "format", "json").lower()
        params = _resolve_env_placeholders(getattr(spec, "params", {}))
        headers = _resolve_env_placeholders(getattr(spec, "headers", {}))
        payload = _resolve_env_placeholders(getattr(spec, "payload", None))
        timeout = getattr(spec, "timeout", 15)
        records_path = getattr(spec, "records_path", None)
        timestamp_key = getattr(spec, "timestamp_key", "timestamp")
        value_key = getattr(spec, "value_key", None)
        value_name = getattr(spec, "value_name", name)
        rename_map = getattr(spec, "rename", {}) or {}

        try:
            response = sess.request(
                method,
                url,
                params=params,
                headers=headers,
                json=payload if method in {"POST", "PUT", "PATCH"} else None,
                timeout=timeout,
            )
            response.raise_for_status()
            if fmt == "json":
                body = response.json()
                records = _navigate_path(body, records_path)
            elif fmt == "csv":
                records = pd.read_csv(io.StringIO(response.text))
            else:
                raise ValueError(f"Unsupported format '{fmt}'")
            frame = _ensure_dataframe(records)
            if frame.empty:
                results.append(
                    ExternalSourceResult(
                        name=name,
                        status="empty",
                        rows=0,
                        message="No rows returned",
                    )
                )
                continue
            if rename_map:
                frame = frame.rename(columns=rename_map)
            ts_col = rename_map.get(timestamp_key, timestamp_key)
            if ts_col not in frame.columns:
                raise KeyError(f"Timestamp column '{timestamp_key}' missing in {name}")
            frame = frame.copy()
            frame["Timestamp"] = _coerce_datetime(frame[ts_col])
            if value_key:
                value_col = rename_map.get(value_key, value_key)
                if value_col not in frame.columns:
                    raise KeyError(f"Value column '{value_key}' missing in {name}")
                if value_name:
                    frame = frame.rename(columns={value_col: value_name})
            frame = frame.dropna(subset=["Timestamp"])
            columns = ["Timestamp"] + [c for c in frame.columns if c != "Timestamp"]
            frame = frame.loc[:, columns]
            frames[name] = frame
            results.append(
                ExternalSourceResult(
                    name=name,
                    status="ok",
                    rows=len(frame),
                )
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed loading external context %s: %s", name, exc)
            results.append(
                ExternalSourceResult(
                    name=name,
                    status="error",
                    rows=0,
                    message=str(exc),
                )
            )

    if not frames:
        return data, results

    def _merge(target: pd.DataFrame) -> pd.DataFrame:
        merged = target
        for name, frame in frames.items():
            merged = merged.merge(frame, on="Timestamp", how=join_type)
        return merged

    if hasattr(data, "apply_chunk"):
        data.apply_chunk(_merge, copy=False)
        meta = getattr(data, "metadata", {})
        if isinstance(meta, dict):
            meta["external_context"] = [r.as_dict() for r in results]
    else:
        data = _merge(data)

    return data, results


__all__ = ["augment_with_external_context", "ExternalSourceResult"]
