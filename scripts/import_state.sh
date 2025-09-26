#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <archive> [--include PATH]" >&2
  exit 1
fi

ARCHIVE="$1"
shift

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
INCLUDES=("data" "config.yaml")
MAX_SIZE=${MAX_ARCHIVE_SIZE:-1073741824}

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.13)}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python3.13 is required to import checkpoints" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include)
      INCLUDES+=("$2"); shift 2 ;;
    *)
      echo "Usage: $0 <archive> [--include PATH]" >&2
      exit 1 ;;
  esac
done

size=$(stat -c%s "$ARCHIVE")
if [ "$size" -gt "$MAX_SIZE" ]; then
  echo "Archive too large" >&2
  exit 1
fi

tmpdir=$(mktemp -d)

tar -tzf "$ARCHIVE" | while read -r entry; do
  [ -z "$entry" ] && continue
  if [[ "$entry" = /* || "$entry" = *..* ]]; then
    echo "Invalid entry in archive: $entry" >&2
    rm -rf "$tmpdir"
    exit 1
  fi
  base="${entry%%/*}"
  allowed=false
  for inc in "${INCLUDES[@]}"; do
    if [[ "$base" == "$inc" ]]; then
      allowed=true
      break
    fi
  done
  if [ "$allowed" = false ]; then
    echo "Unexpected top-level path: $entry" >&2
    rm -rf "$tmpdir"
    exit 1
  fi
done

tar -xzf "$ARCHIVE" -C "$tmpdir"

systemctl stop mt5bot 2>/dev/null || true

for inc in "${INCLUDES[@]}"; do
  src="$tmpdir/$inc"
  if [ -e "$src" ]; then
    if [ "$inc" = "config.yaml" ]; then
      mv "$src" "$REPO_ROOT/config.yaml"
    else
      rm -rf "/$inc"
      mv "$src" "/$inc"
    fi
  fi
done

rm -rf "$tmpdir"

"${PYTHON_BIN}" - <<'PY'
from state_manager import load_latest_checkpoint
ckpt = load_latest_checkpoint()
if ckpt:
    print(f"Loaded checkpoint step {ckpt[0]}")
else:
    print("No checkpoint found")
PY
