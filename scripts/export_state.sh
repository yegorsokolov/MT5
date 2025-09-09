#!/bin/bash
set -euo pipefail

TS=$(date +"%Y%m%d-%H%M%S")
ARCHIVE_DIR=/data/checkpoints
mkdir -p "$ARCHIVE_DIR"
ARCHIVE="$ARCHIVE_DIR/project_state_$TS.tar.gz"
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Store /data and the repository config at the archive root so it can be
# restored without manual tweaks on another server.
tar -czf "$ARCHIVE" -C / data -C "$REPO_ROOT" config.yaml

echo "$ARCHIVE"
