from __future__ import annotations

"""Centralised metrics aggregation for remote backends.

This module provides a thin wrapper around InfluxDB and Prometheus remote
storage endpoints. It exposes convenience functions mirroring the old
``metrics_store`` interface so existing modules can transparently push metrics
without dealing with backend specifics. Remote endpoints are configured via
environment variables and failures are logged but otherwise ignored to avoid
impacting the main application flow.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import time
import logging
import io

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class MetricsAggregator:
    """Push metrics to remote stores and query historical data."""

    influx_url: str | None = None
    influx_token: str | None = None
    influx_org: str | None = None
    influx_bucket: str | None = None
    prom_push_url: str | None = None
    prom_query_url: str | None = None
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:  # pragma: no cover - simple assignment
        self.influx_url = self.influx_url or os.getenv("INFLUXDB_URL")
        self.influx_token = self.influx_token or os.getenv("INFLUXDB_TOKEN")
        self.influx_org = self.influx_org or os.getenv("INFLUXDB_ORG")
        self.influx_bucket = self.influx_bucket or os.getenv("INFLUXDB_BUCKET")
        self.prom_push_url = self.prom_push_url or os.getenv("PROM_PUSH_URL")
        self.prom_query_url = self.prom_query_url or os.getenv("PROM_QUERY_URL")

    # ------------------------------------------------------------------
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        """Send a single metric sample to configured backends."""

        tags = tags or {}
        timestamp = int(time.time() * 1_000_000_000)  # ns

        # InfluxDB line protocol
        if self.influx_url and self.influx_bucket and self.influx_org:
            tag_str = ",".join(f"{k}={v}" for k, v in tags.items())
            line = f"{name}{',' + tag_str if tag_str else ''} value={float(value)} {timestamp}"
            params = {"bucket": self.influx_bucket, "org": self.influx_org, "precision": "ns"}
            headers = {}
            if self.influx_token:
                headers["Authorization"] = f"Token {self.influx_token}"
            try:
                requests.post(
                    f"{self.influx_url}/api/v2/write",
                    params=params,
                    data=line.encode("utf-8"),
                    headers=headers,
                    timeout=2,
                )
            except Exception:  # pragma: no cover - best effort
                logger.debug("InfluxDB push failed", exc_info=True)

        # Prometheus Pushgateway / remote write in text exposition format
        if self.prom_push_url:
            try:
                labels = ",".join(f'{k}="{v}"' for k, v in tags.items())
                metric = f"{name}{{{labels}}} {float(value)}" if labels else f"{name} {float(value)}"
                requests.post(self.prom_push_url, data=metric.encode("utf-8"), timeout=2)
            except Exception:  # pragma: no cover - best effort
                logger.debug("Prometheus push failed", exc_info=True)

    # ------------------------------------------------------------------
    def query_metrics(
        self,
        name: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Fetch metrics from remote stores as a dataframe.

        Attempts InfluxDB first, then Prometheus. Returns an empty dataframe if
        no backend is reachable.
        """

        tags = tags or {}

        # InfluxDB query
        if self.influx_url and self.influx_org and self.influx_bucket:
            start_expr = pd.Timestamp(start).isoformat() if start else "-30d"
            end_expr = pd.Timestamp(end).isoformat() if end else None
            filter_parts = [f'r._measurement == "{name}"']
            for k, v in tags.items():
                filter_parts.append(f'r.{k} == "{v}"')
            flt = " and ".join(filter_parts)
            range_clause = f"|> range(start: {start_expr}"
            if end_expr:
                range_clause += f", stop: {end_expr}"
            range_clause += ")"
            flux = f'from(bucket: "{self.influx_bucket}") {range_clause} |> filter(fn: (r) => {flt})'
            headers = {"Accept": "application/csv"}
            if self.influx_token:
                headers["Authorization"] = f"Token {self.influx_token}"
            try:
                resp = requests.post(
                    f"{self.influx_url}/api/v2/query",
                    params={"org": self.influx_org},
                    data=flux,
                    headers=headers,
                    timeout=5,
                )
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                if "_time" in df and "_value" in df:
                    return df.rename(columns={"_time": "timestamp", "_value": "value"})[["timestamp", "value"]]
            except Exception:  # pragma: no cover - best effort
                logger.debug("InfluxDB query failed", exc_info=True)

        # Prometheus query_range
        if self.prom_query_url:
            try:
                start_sec = int(pd.Timestamp(start).timestamp()) if start else int(time.time() - 30 * 86400)
                end_sec = int(pd.Timestamp(end).timestamp()) if end else int(time.time())
                label = ",".join(f'{k}="{v}"' for k, v in tags.items())
                promql = f"{name}{{{label}}}" if label else name
                params = {
                    "query": promql,
                    "start": start_sec,
                    "end": end_sec,
                    "step": max(int((end_sec - start_sec) / 100), 1),
                }
                resp = requests.get(
                    f"{self.prom_query_url}/api/v1/query_range", params=params, timeout=5
                )
                resp.raise_for_status()
                data = resp.json().get("data", {}).get("result", [])
                if data:
                    values = data[0].get("values", [])
                    df = pd.DataFrame(values, columns=["timestamp", "value"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df["value"] = df["value"].astype(float)
                    return df
            except Exception:  # pragma: no cover - best effort
                logger.debug("Prometheus query failed", exc_info=True)

        return pd.DataFrame(columns=["timestamp", "value"])

    # ------------------------------------------------------------------
    def log_retrain_outcome(self, model: str, status: str) -> None:
        """Helper for logging retrain success/failure."""
        self.record_metric(
            "retrain_outcome",
            1.0 if status == "success" else 0.0,
            {"model": model, "status": status},
        )

    # Convenience wrappers -------------------------------------------------


def _get_default() -> MetricsAggregator:
    global _DEFAULT_AGG
    try:
        return _DEFAULT_AGG
    except NameError:
        _DEFAULT_AGG = MetricsAggregator()
        return _DEFAULT_AGG


def record_metric(name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
    _get_default().record_metric(name, value, tags)


def query_metrics(
    name: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tags: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    return _get_default().query_metrics(name, start=start, end=end, tags=tags)


def log_retrain_outcome(model: str, status: str) -> None:
    _get_default().log_retrain_outcome(model, status)


def model_cache_hit() -> None:
    record_metric("model_cache_hits", 1.0)


def model_unload() -> None:
    record_metric("model_unloads", 1.0)


__all__ = [
    "MetricsAggregator",
    "record_metric",
    "query_metrics",
    "log_retrain_outcome",
    "model_cache_hit",
    "model_unload",
]
