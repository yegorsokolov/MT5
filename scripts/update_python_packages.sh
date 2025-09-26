#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "${PROJECT_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to update dependencies" >&2
    exit 1
fi

echo "Ensuring pip itself is up to date..."
python3 -m pip install --upgrade pip

REQ_FILE="requirements.txt"

if [[ -f "${REQ_FILE}" ]]; then
    echo "Synchronising packages from ${REQ_FILE}..."
    python3 -m pip install --upgrade --upgrade-strategy eager -r "${REQ_FILE}"
else
    echo "Warning: ${REQ_FILE} not found; skipping dependency synchronisation." >&2
fi

echo "Validating dependency consistency with pip check..."
python3 -m pip check

echo "Python dependencies are up to date."
