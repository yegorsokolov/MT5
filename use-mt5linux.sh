#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
# shellcheck source=scripts/_mt5linux_env.sh
source "${PROJECT_ROOT}/scripts/_mt5linux_env.sh"

if ! refresh_mt5linux_venv "${MT5LINUX_BOOTSTRAP_PYTHON:-}"; then
  echo "[error] Failed to prepare mt5linux auxiliary environment at $MT5LINUX_VENV_PATH" >&2
  exit 1
fi

if [[ ! -d "$MT5LINUX_VENV_PATH" ]]; then
  echo "[error] mt5linux auxiliary environment missing at $MT5LINUX_VENV_PATH" >&2
  exit 1
fi

source "$MT5LINUX_VENV_PATH/bin/activate"
echo "[info] mt5linux env active: $(python -V 2>&1)"
exec "$SHELL" -l
