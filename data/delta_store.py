"""Utilities for tracking dataset deltas.

The :class:`DeltaStore` keeps a small JSON index mapping ingested file
paths to their last known SHA256 hash and number of processed rows.  When a
file grows, only the newly appended records are written to a corresponding
``*.delta`` log next to the source file.  These delta logs can then be
applied to cached datasets so repeated runs only process new data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from utils.data_backend import get_dataframe_module
from .versioning import compute_hash

pd = get_dataframe_module()


class DeltaStore:
    """Track ingested files and persist change logs for new rows.

    Parameters
    ----------
    state_path: Path, optional
        Location of the JSON file maintaining metadata about ingested files.
        Defaults to ``data/delta_state.json`` under the package directory.
    """

    def __init__(self, state_path: Path | None = None) -> None:
        if state_path is None:
            state_path = Path(__file__).resolve().parent / "delta_state.json"
        self.state_path = state_path
        if state_path.exists():
            try:
                self.state: Dict[str, Dict[str, Any]] = json.loads(
                    state_path.read_text()
                )
            except Exception:
                self.state = {}
        else:
            self.state = {}

    # ------------------------------------------------------------------
    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True))

    # ------------------------------------------------------------------
    def ingest(self, path: Path, df) -> pd.DataFrame:  # type: ignore[override]
        """Record ``df`` loaded from ``path`` and return newly appended rows.

        Parameters
        ----------
        path: Path
            Source file path from which ``df`` was loaded.
        df: pd.DataFrame
            Full dataframe currently present on disk.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the rows that were appended since the
            last ingestion.  On the first ingestion the full dataframe is
            returned but no ``*.delta`` log is written.
        """

        path = Path(path)
        key = str(path.resolve())
        new_hash = compute_hash(path)
        prev = self.state.get(key)

        delta_df = df
        write_delta = False
        if prev:
            prev_rows = int(prev.get("rows", 0))
            if prev_rows < len(df):
                delta_df = df.iloc[prev_rows:].copy()
                write_delta = True
            else:
                delta_df = pd.DataFrame(columns=df.columns)
        else:
            # First time seeing this file – the entire dataframe is "new"
            # but we don't persist a delta log for the initial load.
            delta_df = df.copy()

        if write_delta and not delta_df.empty:
            delta_path = path.with_suffix(path.suffix + ".delta")
            delta_path.parent.mkdir(parents=True, exist_ok=True)
            if delta_path.exists():
                delta_df.to_csv(
                    delta_path,
                    mode="a",
                    header=False,
                    index=False,
                    date_format="%Y%m%d %H:%M:%S:%f",
                )
            else:
                delta_df.to_csv(
                    delta_path,
                    index=False,
                    date_format="%Y%m%d %H:%M:%S:%f",
                )

        self.state[key] = {"hash": new_hash, "rows": len(df)}
        self._save()
        return delta_df


__all__ = ["DeltaStore"]
