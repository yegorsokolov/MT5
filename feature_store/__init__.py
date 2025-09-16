from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

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


def _default_code_hash(df: pd.DataFrame) -> str:
    """Generate a deterministic hash describing the dataframe schema."""

    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, df.columns)).encode("utf-8"))
    hasher.update("|".join(str(df[col].dtype) for col in df.columns).encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))
    return hasher.hexdigest()


def register_feature(
    version: str,
    df: pd.DataFrame,
    metadata: Dict[str, Any] | None = None,
    *,
    code_hash: str | None = None,
) -> None:
    """Persist ``df`` under ``version`` and record ``metadata`` and ``code_hash``."""
    STORE_DIR.mkdir(exist_ok=True)
    data_path = STORE_DIR / f"{version}.pkl"
    df.to_pickle(data_path)

    meta = dict(metadata or {})
    if code_hash is None:
        code_hash = meta.pop("code_hash", None)
    else:
        meta.pop("code_hash", None)
    if code_hash is None:
        code_hash = _default_code_hash(df)

    index = _load_index()
    index[version] = {
        "rows": len(df),
        "columns": [str(col) for col in df.columns],
        "code_hash": code_hash,
        **meta,
    }
    index["latest"] = version
    _save_index(index)


def load_feature(
    version: str, *, with_metadata: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load feature ``version`` from the store.

    Parameters
    ----------
    version:
        Identifier of the stored feature artifact.
    with_metadata:
        When ``True`` return both the dataframe and its metadata.
    """

    data_path = STORE_DIR / f"{version}.pkl"
    if not data_path.exists():
        raise FileNotFoundError(version)
    df = pd.read_pickle(data_path)
    if not with_metadata:
        return df
    metadata = dict(_load_index().get(version, {}))
    return df, metadata


def latest_version() -> str | None:
    """Return the most recently registered version hash."""
    return _load_index().get("latest")


def list_versions() -> Dict[str, Any]:
    """Return mapping of version hashes to their metadata."""
    index = _load_index()
    return {
        key: dict(value)
        for key, value in index.items()
        if key != "latest"
    }


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
