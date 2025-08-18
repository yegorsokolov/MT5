"""Simple feature caching using DuckDB.

The cache stores feature DataFrames keyed by a combination of
symbol, window size and additional parameters.  Each entry also records
an associated hash of the raw input data so cached features are
invalidated automatically when the underlying data changes.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import os

import duckdb
import pandas as pd
import requests
from utils.secret_manager import SecretManager
from utils.resource_monitor import monitor


class FeatureStore:
    """Persist and retrieve feature DataFrames using DuckDB.

    When ``service_url`` is provided, the store will attempt to fetch feature
    data from that remote service before falling back to the local DuckDB cache.
    Any locally computed features are uploaded back to the service. This allows
    sharing feature computations across workers while still operating if the
    service becomes unavailable.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        service_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tls_cert: Optional[str] = None,
    ) -> None:
        if path is None:
            path = Path(__file__).resolve().parent / "data" / "feature_store.duckdb"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.service_url = service_url or os.getenv("FEATURE_SERVICE_URL")
        sm = SecretManager()
        self.api_key = api_key or sm.get_secret("FEATURE_SERVICE_API_KEY")
        self.tls_cert = tls_cert or os.getenv("FEATURE_SERVICE_CA_CERT")

    # ------------------------------------------------------------------
    def _key(self, symbol: str, window: int, params: Dict[str, Any]) -> str:
        param_json = json.dumps(params, sort_keys=True, default=str)
        raw = f"{symbol}_{window}_{param_json}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    def load(
        self,
        symbol: str,
        window: int,
        params: Dict[str, Any],
        raw_hash: str,
    ) -> Optional[pd.DataFrame]:
        """Return cached features if available and up-to-date."""
        usage = getattr(monitor, "latest_usage", {})
        disk_pressure = usage.get("disk_read", 0) + usage.get("disk_write", 0)
        net_pressure = usage.get("net_rx", 0) + usage.get("net_tx", 0)
        if (
            self.service_url
            and disk_pressure > 50 * 1024 * 1024
            and net_pressure < 10 * 1024 * 1024
            and params.get("start")
            and params.get("end")
        ):
            remote = self.fetch_remote(symbol, params["start"], params["end"])
            if remote is not None:
                return remote

        if not self.path.exists():
            return None

        conn = duckdb.connect(self.path.as_posix())
        key = self._key(symbol, window, params)
        table_name = f"features_{key}"
        tables = {t[0] for t in conn.execute("PRAGMA show_tables").fetchall()}
        if table_name not in tables:
            conn.close()
            return None

        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key VARCHAR PRIMARY KEY, raw_hash VARCHAR)"
        )
        row = conn.execute(
            "SELECT raw_hash FROM metadata WHERE key = ?", [key]
        ).fetchone()
        if not row or row[0] != raw_hash:
            conn.close()
            return None

        df = conn.execute(f'SELECT * FROM "{table_name}"').fetch_df()
        conn.close()
        return df

    # ------------------------------------------------------------------
    def load_any(
        self, symbol: str, window: int, params: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Return cached features ignoring any raw data hash.

        This helper is used by delta-aware feature generation which only
        appends new feature rows on top of existing cached data.  The caller
        is responsible for ensuring the underlying raw data has not changed
        except for new rows appended at the end.
        """

        if not self.path.exists():
            return None

        conn = duckdb.connect(self.path.as_posix())
        key = self._key(symbol, window, params)
        table_name = f"features_{key}"
        tables = {t[0] for t in conn.execute("PRAGMA show_tables").fetchall()}
        if table_name not in tables:
            conn.close()
            return None

        df = conn.execute(f'SELECT * FROM "{table_name}"').fetch_df()
        conn.close()
        return df

    # ------------------------------------------------------------------
    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        window: int,
        params: Dict[str, Any],
        raw_hash: str,
    ) -> None:
        """Persist ``df`` in the cache."""

        conn = duckdb.connect(self.path.as_posix())
        key = self._key(symbol, window, params)
        table_name = f"features_{key}"
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key VARCHAR PRIMARY KEY, raw_hash VARCHAR)"
        )
        conn.execute("DELETE FROM metadata WHERE key = ?", [key])
        conn.execute("INSERT INTO metadata VALUES (?, ?)", [key, raw_hash])
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.register("feat_df", df)
        conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM feat_df')
        conn.close()

    # ------------------------------------------------------------------
    def get_or_set(
        self,
        symbol: str,
        window: int,
        params: Dict[str, Any],
        raw_hash: str,
        compute_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Return cached features or compute and cache them.

        This helper avoids repetition when a feature generation step can be
        cached.  ``compute_fn`` is only executed when no valid cache entry is
        available.
        """

        cached = self.load(symbol, window, params, raw_hash)
        if cached is not None:
            return cached
        df = compute_fn()
        self.save(df, symbol, window, params, raw_hash)
        return df

    # ------------------------------------------------------------------
    def invalidate(self, symbol: str, window: int, params: Dict[str, Any]) -> None:
        """Remove a cached entry if it exists."""

        if not self.path.exists():
            return
        conn = duckdb.connect(self.path.as_posix())
        key = self._key(symbol, window, params)
        table_name = f"features_{key}"
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key VARCHAR PRIMARY KEY, raw_hash VARCHAR)"
        )
        conn.execute("DELETE FROM metadata WHERE key = ?", [key])
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.close()

    # ------------------------------------------------------------------
    def fetch_remote(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Try retrieving features from the remote service.

        Parameters are passed directly as query arguments to the service. If the
        service is unreachable or returns a non-200 status code, ``None`` is
        returned so callers can fallback to local computation.
        """

        if not self.service_url:
            return None
        url = f"{self.service_url.rstrip('/')}/features/{symbol}"
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            resp = requests.get(
                url,
                params={"start": start, "end": end},
                headers=headers,
                timeout=10,
                verify=self.tls_cert or True,
            )
            if resp.status_code == 200:
                return pd.DataFrame(resp.json())
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def upload_remote(self, df: pd.DataFrame, symbol: str, start: str, end: str) -> None:
        """Upload locally computed features to the remote service.

        Failures are silently ignored so that training can continue even if the
        service is temporarily unavailable.
        """

        if not self.service_url:
            return
        url = f"{self.service_url.rstrip('/')}/features/{symbol}"
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            requests.post(
                url,
                params={"start": start, "end": end},
                headers=headers,
                json=df.to_dict(orient="records"),
                timeout=10,
                verify=self.tls_cert or True,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def get_features(
        self,
        symbol: str,
        start: str,
        end: str,
        compute_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Fetch features from the service or compute locally.

        The remote service is queried first; if it responds with a valid
        DataFrame the result is cached locally and returned. Otherwise
        ``compute_fn`` is executed and the resulting features are saved both
        locally and uploaded to the service. This provides a transparent
        fallback to local computation when the service cannot be reached.
        """

        params = {"start": start, "end": end}
        df = self.fetch_remote(symbol, start, end)
        if df is not None:
            self.save(df, symbol, 0, params, raw_hash="remote")
            return df
        df = compute_fn()
        self.save(df, symbol, 0, params, raw_hash="remote")
        self.upload_remote(df, symbol, start, end)
        return df


__all__ = ["FeatureStore"]
