from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Directory for stored features and metadata
STORE_DIR = Path(__file__).resolve().parent / "store"
INDEX_FILE = STORE_DIR / "index.json"


def _load_index() -> Dict[str, Any]:
    STORE_DIR.mkdir(exist_ok=True)
    if INDEX_FILE.exists():
        with INDEX_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_index(index: Dict[str, Any]) -> None:
    STORE_DIR.mkdir(exist_ok=True)
    with INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def register_feature(
    version: str, df: pd.DataFrame, metadata: Dict[str, Any] | None = None
) -> None:
    """Persist ``df`` under ``version`` and record ``metadata``."""
    STORE_DIR.mkdir(exist_ok=True)
    data_path = STORE_DIR / f"{version}.pkl"
    df.to_pickle(data_path)
    index = _load_index()
    index[version] = {"rows": len(df), **(metadata or {})}
    index["latest"] = version
    _save_index(index)


def load_feature(version: str) -> pd.DataFrame:
    """Load feature ``version`` from the store."""
    data_path = STORE_DIR / f"{version}.pkl"
    if not data_path.exists():
        raise FileNotFoundError(version)
    return pd.read_pickle(data_path)


def latest_version() -> str | None:
    """Return the most recently registered version hash."""
    return _load_index().get("latest")


def list_versions() -> Dict[str, Any]:
    """Return mapping of version hashes to their metadata."""
    index = _load_index().copy()
    index.pop("latest", None)
    return index


def purge_version(version: str) -> None:
    """Remove ``version`` and its metadata from the store."""
    data_path = STORE_DIR / f"{version}.pkl"
    if data_path.exists():
        data_path.unlink()
    index = _load_index()
    index.pop(version, None)
    if index.get("latest") == version:
        index.pop("latest", None)
    _save_index(index)


def request_indicator(version: str) -> list[str]:
    """Return available indicator columns for ``version``."""

    df = load_feature(version)
    return list(df.columns)
