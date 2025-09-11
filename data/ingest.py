from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import List

try:
    from .versioning import compute_hash
except Exception:  # pragma: no cover - import without package context
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "versioning", Path(__file__).resolve().parent / "versioning.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    compute_hash = module.compute_hash  # type: ignore[attr-defined]

VERSIONS_FILE = Path(__file__).resolve().parents[1] / "data_versions.json"


def _load_versions() -> dict:
    if VERSIONS_FILE.exists():
        try:
            data = json.loads(VERSIONS_FILE.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data.setdefault("lineage", {})
    return data


def _feature_hash(rows: List[List[str]]) -> str:
    """Compute SHA256 hash for a list of CSV rows."""
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def record_lineage(raw_path: Path, rows: List[List[str]]) -> None:
    """Record SHA256 hashes for the raw input and derived features."""
    raw_hash = compute_hash(raw_path)
    feature_hash = _feature_hash(rows)
    data = _load_versions()
    data["lineage"][str(raw_path.resolve())] = {
        "raw": raw_hash,
        "features": feature_hash,
    }
    VERSIONS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))


def ingest(raw_path: Path, record: bool = False) -> List[List[str]]:
    """Read CSV rows from ``raw_path`` and optionally record lineage."""
    with raw_path.open() as f:
        rows = list(csv.reader(f))
    if record:
        record_lineage(raw_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw data and record lineage hashes"
    )
    parser.add_argument("raw_path", help="Path to raw CSV file")
    parser.add_argument(
        "--record-lineage",
        action="store_true",
        help="Compute and store lineage hashes in data_versions.json",
    )
    args = parser.parse_args()
    ingest(Path(args.raw_path), record=args.record_lineage)


if __name__ == "__main__":
    main()
