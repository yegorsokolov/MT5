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
import time

import duckdb
import pandas as pd
from analytics.metrics_store import record_metric
from services.worker_manager import get_worker_manager
from analysis.data_lineage import log_lineage


class FeatureStore:
    """Persist and retrieve feature DataFrames using DuckDB.

    The store now operates entirely on the local DuckDB cache. Remote
    feature APIs have been removed from the supported stack; host the
    archived services separately if you still require them.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        service_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tls_cert: Optional[str] = None,
        memory_size: int = 128,
        worker_url: Optional[str] = None,
        worker_retries: int = 3,
    ) -> None:
        if path is None:
            path = Path(__file__).resolve().parent / "data" / "feature_store.duckdb"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        disallowed = {
            "service_url": service_url or os.getenv("FEATURE_SERVICE_URL"),
            "api_key": api_key,
            "tls_cert": tls_cert or os.getenv("FEATURE_SERVICE_CA_CERT"),
            "worker_url": worker_url or os.getenv("FEATURE_WORKER_URL"),
        }
        if any(disallowed.values()):
            raise RuntimeError(
                "Remote feature services were removed from the MT5 toolkit. "
                "Run the archived FastAPI or worker components separately if "
                "you still depend on them."
            )
        self.memory_size = memory_size
        self._memory: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = {"memory": 0, "disk": 0}
        self._totals = {"memory": 0, "disk": 0}

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

        run_id = df.attrs.get("run_id", "unknown")
        raw_file = df.attrs.get("source", "unknown")
        for col in df.columns:
            log_lineage(run_id, raw_file, "feature_store.save", col)

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
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def get_features(
        self,
        symbol: str,
        start: str,
        end: str,
        compute_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Fetch features using tiered caching.

        The lookup order is in-memory LRU cache followed by the local
        DuckDB cache. Misses fall back to ``compute_fn`` and the result is
        cached asynchronously for future requests. Tier hit ratios are
        reported via :func:`analytics.metrics_store.record_metric`.
        """

        start_t = time.perf_counter()
        try:
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
            df = self.load(symbol, 0, params, raw_hash="local")
            if df is not None:
                self._hits["disk"] += 1
                threading.Thread(
                    target=self._cache_memory, args=(key, df), daemon=True
                ).start()
                self._record_metrics()
                return df

            # Compute fallback ---------------------------------------------
            df = compute_fn()
            threading.Thread(
                target=self.save, args=(df, symbol, 0, params, "local"), daemon=True
            ).start()
            threading.Thread(
                target=self._cache_memory, args=(key, df), daemon=True
            ).start()
            self._record_metrics()
            return df
        finally:
            latency = time.perf_counter() - start_t
            get_worker_manager().record_request("feature_store", latency)

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
