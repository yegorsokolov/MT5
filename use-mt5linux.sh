#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
# shellcheck source=scripts/_mt5linux_env.sh
source "${PROJECT_ROOT}/scripts/_mt5linux_env.sh"

if [[ ! -d "$MT5LINUX_VENV_PATH" ]]; then
  refresh_mt5linux_venv "${MT5LINUX_BOOTSTRAP_PYTHON:-}"
fi

source "$MT5LINUX_VENV_PATH/bin/activate"
echo "[info] mt5linux env active: $(python -V 2>&1)"
exec "$SHELL" -l
