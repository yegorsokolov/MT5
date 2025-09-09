#!/bin/bash
set -euo pipefail

TS=$(date +"%Y%m%d-%H%M%S")
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARCHIVE="$REPO_ROOT/project_state_$TS.tar.gz"

INCLUDES=("/data" "$REPO_ROOT/config.yaml")
EXCLUDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include)
      INCLUDES+=("$2"); shift 2 ;;
    --exclude)
      EXCLUDES+=("--exclude=$2"); shift 2 ;;
    *)
      echo "Usage: $0 [--include PATH] [--exclude PATTERN]" >&2
      exit 1 ;;
  esac
done

tar_args=("-czf" "$ARCHIVE" "${EXCLUDES[@]}")

for p in "${INCLUDES[@]}"; do
  if [[ -e "$p" ]]; then
    abs=$(readlink -f "$p")
    dir=$(dirname "$abs")
    base=$(basename "$abs")
    tar_args+=("-C" "$dir" "$base")
  else
    echo "Warning: $p not found, skipping" >&2
  fi
done

tar "${tar_args[@]}"

echo "$ARCHIVE"
