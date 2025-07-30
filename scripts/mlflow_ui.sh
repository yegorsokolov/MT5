#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
mlflow ui --backend-store-uri "${REPO_ROOT}/logs/mlruns" "$@"
