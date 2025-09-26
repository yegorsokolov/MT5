#!/usr/bin/env bash
set -euo pipefail

# Run the environment diagnostics and render the manual checklist.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd "${PROJECT_ROOT}"

python -m utils.environment --no-auto-install "$@"
