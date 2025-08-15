#!/usr/bin/env bash
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${REPO_DIR}/logs"
touch "${REPO_DIR}/logs/app.log"
