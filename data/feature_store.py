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

import duckdb
import pandas as pd


class FeatureStore:
    """Persist and retrieve feature DataFrames using DuckDB."""

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parent / "data" / "feature_store.duckdb"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

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


__all__ = ["FeatureStore"]
