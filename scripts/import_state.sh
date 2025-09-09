#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <archive>" >&2
  exit 1
fi

ARCHIVE="$1"

tar -xzf "$ARCHIVE" -C /

python3 - <<'PY'
from state_manager import load_latest_checkpoint
ckpt = load_latest_checkpoint()
if ckpt:
    print(f"Loaded checkpoint step {ckpt[0]}")
else:
    print("No checkpoint found")
PY
