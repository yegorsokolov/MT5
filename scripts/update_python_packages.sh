#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}" 
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in python3 python3.10 python3.11 python3.12; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "A supported python3 interpreter is required to update dependencies." >&2
    exit 1
fi

PYTHON_MAJOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.minor)')
if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 10) )); then
    echo "Python ${PYTHON_MAJOR}.${PYTHON_MINOR} is not supported. Please use Python 3.10 or newer." >&2
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
