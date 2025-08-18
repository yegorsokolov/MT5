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
from collections import OrderedDict
import threading

import duckdb
import pandas as pd
import requests
from analytics.metrics_store import record_metric
try:  # pragma: no cover - optional dependency in tests
    from utils.secret_manager import SecretManager
except Exception:  # pragma: no cover
    class SecretManager:  # type: ignore
        def get_secret(self, *_args: Any, **_kwargs: Any) -> None:
            return None
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
        memory_size: int = 128,
    ) -> None:
        if path is None:
            path = Path(__file__).resolve().parent / "data" / "feature_store.duckdb"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.service_url = service_url or os.getenv("FEATURE_SERVICE_URL")
        sm = SecretManager()
        self.api_key = api_key or sm.get_secret("FEATURE_SERVICE_API_KEY")
        self.tls_cert = tls_cert or os.getenv("FEATURE_SERVICE_CA_CERT")
        self.memory_size = memory_size
        self._memory: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = {"memory": 0, "disk": 0, "remote": 0}
        self._totals = {"memory": 0, "disk": 0, "remote": 0}

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
        """Fetch features using tiered caching.

        The lookup order is in-memory LRU cache, local DuckDB cache and
        finally the remote feature service.  Misses fall back to
        ``compute_fn``.  When a lower tier serves the request, upper tiers are
        populated asynchronously so subsequent calls are faster.  Tier hit
        ratios are reported via :func:`analytics.metrics_store.record_metric`.
        """

        params = {"start": start, "end": end}
        key = self._key(symbol, 0, params)

        # Memory tier --------------------------------------------------
        self._totals["memory"] += 1
        with self._lock:
            if key in self._memory:
                df = self._memory.pop(key)
                self._memory[key] = df
                self._hits["memory"] += 1
                self._record_metrics()
                return df

        # Disk tier ----------------------------------------------------
        self._totals["disk"] += 1
        df = self.load(symbol, 0, params, raw_hash="remote")
        if df is not None:
            self._hits["disk"] += 1
            threading.Thread(
                target=self._cache_memory, args=(key, df), daemon=True
            ).start()
            self._record_metrics()
            return df

        # Remote tier --------------------------------------------------
        if self.service_url:
            self._totals["remote"] += 1
            df = self.fetch_remote(symbol, start, end)
            if df is not None:
                self._hits["remote"] += 1
                threading.Thread(
                    target=self.save,
                    args=(df, symbol, 0, params, "remote"),
                    daemon=True,
                ).start()
                threading.Thread(
                    target=self._cache_memory, args=(key, df), daemon=True
                ).start()
                self._record_metrics()
                return df

        # Compute fallback ---------------------------------------------
        df = compute_fn()
        threading.Thread(
            target=self.save, args=(df, symbol, 0, params, "remote"), daemon=True
        ).start()
        threading.Thread(
            target=self._cache_memory, args=(key, df), daemon=True
        ).start()
        threading.Thread(
            target=self.upload_remote, args=(df, symbol, start, end), daemon=True
        ).start()
        self._record_metrics()
        return df

    # ------------------------------------------------------------------
    def _cache_memory(self, key: str, df: pd.DataFrame) -> None:
        """Store ``df`` in the in-memory LRU cache."""

        with self._lock:
            self._memory[key] = df
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_size:
                self._memory.popitem(last=False)

    # ------------------------------------------------------------------
    def _record_metrics(self) -> None:
        """Record current hit ratios for each tier."""

        for tier, hits in self._hits.items():
            total = self._totals[tier]
            if total:
                ratio = hits / total
                record_metric(
                    "feature_store_hit_ratio", ratio, tags={"tier": tier}
                )


__all__ = ["FeatureStore"]
