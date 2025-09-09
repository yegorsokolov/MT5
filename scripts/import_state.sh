#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <archive>" >&2
  exit 1
fi

ARCHIVE="$1"

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Stop any running services that might lock files
systemctl stop mt5bot 2>/dev/null || true

# Remove existing state
rm -rf /data
rm -f "$REPO_ROOT/config.yaml"

# Extract archive: /data goes to root, config.yaml moved to project root
tar -xzf "$ARCHIVE" -C /
mv /config.yaml "$REPO_ROOT/config.yaml"

# Load latest checkpoint
python3 - <<'PY'
from state_manager import load_latest_checkpoint
ckpt = load_latest_checkpoint()
if ckpt:
    print(f"Loaded checkpoint step {ckpt[0]}")
else:
    print("No checkpoint found")
PY
