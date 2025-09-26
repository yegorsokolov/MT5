#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/diagnostics"
mkdir -p "$LOG_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"

# Log current Python packages
FREEZE_FILE="$LOG_DIR/pip_freeze_${TS}.txt"
pip freeze > "$FREEZE_FILE"

export LOG_DIR TS FREEZE_FILE
python - <<'PY'
import os
from pathlib import Path

try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - alerting optional
    def send_alert(msg: str) -> None:
        pass

log_dir = Path(os.environ["LOG_DIR"])
ts = os.environ["TS"]
freeze_file = Path(os.environ["FREEZE_FILE"])
freeze_lines = set(freeze_file.read_text().splitlines())

req_path = Path("requirements.txt")
required: list[str] = []
if req_path.exists():
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        required.append(line)

missing = [r for r in required if r not in freeze_lines]
if missing:
    mismatch_file = log_dir / f"dependency_mismatch_{ts}.log"
    mismatch_file.write_text("\n".join(missing))
    send_alert(
        "Dependency mismatch detected. Run 'pip install -r requirements.txt' to sync."
    )
PY

export DIAG_LOG_DIR="$LOG_DIR"
export DIAG_TS="$TS"
python - <<'PY'
import importlib.util
import os, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("versioning", Path("data/versioning.py"))
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)
compute_hash = module.compute_hash

try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - alerting optional
    def send_alert(msg: str) -> None:
        pass

log_dir = Path(os.environ["DIAG_LOG_DIR"])
ts = os.environ["DIAG_TS"]

baseline_path = Path("data_versions.json")
if not baseline_path.exists():
    sys.exit(0)

baseline = json.loads(baseline_path.read_text())
mismatches: list[str] = []
hashes: dict[str, str] = {}
for path, expected in baseline.items():
    p = Path(path)
    if not p.exists():
        mismatches.append(f"{path}: missing file")
        continue
    actual = compute_hash(p)
    hashes[path] = actual
    if actual != expected:
        mismatches.append(f"{path}: expected {expected}, got {actual}")

(log_dir / f"data_hashes_{ts}.json").write_text(json.dumps(hashes, indent=2))

if mismatches:
    mismatch_file = log_dir / f"data_mismatch_{ts}.log"
    mismatch_file.write_text("\n".join(mismatches))
    send_alert(
        "Data version mismatch detected. Run 'python scripts/check_data_versions.py' to update datasets."
    )
PY

