#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/.mt5linux-venv/bin/activate"
echo "[info] mt5linux env active: $(python -V 2>&1)"
exec "$SHELL" -l
