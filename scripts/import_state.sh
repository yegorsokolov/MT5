#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <archive>" >&2
  exit 1
fi

ARCHIVE="$1"
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Remove existing state to ensure the import fully replaces it.
rm -rf /data
mkdir -p /data
rm -f "$REPO_ROOT/config.yaml"

# Restore /data and config.yaml from the archive.
tar -xzf "$ARCHIVE" -C / data
tar -xzf "$ARCHIVE" -C "$REPO_ROOT" config.yaml

python3 - <<'PY'
try:
    from state_manager import load_latest_checkpoint
    ckpt = load_latest_checkpoint()
    if ckpt:
        print(f"Loaded checkpoint step {ckpt[0]}")
    else:
        print("No checkpoint found")
except Exception as exc:
    print(f"Checkpoint load failed: {exc}")
PY
