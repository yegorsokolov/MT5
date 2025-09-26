#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.13)}"

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python3.13 is required to update dependencies" >&2
    exit 1
fi

echo "Ensuring pip itself is up to date..."
"${PYTHON_BIN}" -m pip install --upgrade pip

REQ_FILE="requirements.txt"

if [[ -f "${REQ_FILE}" ]]; then
    echo "Synchronising packages from ${REQ_FILE}..."
    "${PYTHON_BIN}" -m pip install --upgrade --upgrade-strategy eager -r "${REQ_FILE}"
else
    echo "Warning: ${REQ_FILE} not found; skipping dependency synchronisation." >&2
fi

echo "Validating dependency consistency with pip check..."
"${PYTHON_BIN}" -m pip check

echo "Python dependencies are up to date."
