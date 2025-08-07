"""Compare current dataset hashes with committed versions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

BASELINE_PATH = Path("data_versions.json")
LOG_PATH = Path("logs/data_versions.json")


def main() -> int:
    if not BASELINE_PATH.exists():
        print("No baseline data_versions.json found", file=sys.stderr)
        return 0
    if not LOG_PATH.exists():
        print("No logged data versions found", file=sys.stderr)
        return 0
    baseline = json.loads(BASELINE_PATH.read_text())
    current = json.loads(LOG_PATH.read_text())
    mismatches = []
    for symbol, expected in baseline.items():
        actual = current.get(symbol)
        if actual != expected:
            mismatches.append(f"{symbol}: expected {expected}, got {actual}")
    if mismatches:
        for m in mismatches:
            print(m, file=sys.stderr)
        return 1
    print("Data versions match baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
