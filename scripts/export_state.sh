#!/bin/bash
set -euo pipefail

TS=$(date +"%Y%m%d-%H%M%S")
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARCHIVE="$REPO_ROOT/project_state_$TS.tar.gz"

# Archive /data and config.yaml at the root of the tarball
tar -czf "$ARCHIVE" -C / data -C "$REPO_ROOT" config.yaml

echo "$ARCHIVE"
