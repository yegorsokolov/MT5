#!/bin/bash
set -euo pipefail

TS=$(date +"%Y%m%d-%H%M%S")
ARCHIVE_DIR=/data/checkpoints
mkdir -p "$ARCHIVE_DIR"
ARCHIVE="$ARCHIVE_DIR/project_state_$TS.tar.gz"
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

tar -czf "$ARCHIVE" /data "$REPO_ROOT/config.yaml"

echo "$ARCHIVE"
