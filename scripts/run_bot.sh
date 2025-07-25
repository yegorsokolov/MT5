#!/usr/bin/env bash
set -e

# Root of repo
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
cd "$REPO_ROOT"

MODEL_PATH="model.joblib"
export CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config.yaml}"

# train if model missing or TRAIN_ALWAYS=1
if [ ! -f "$MODEL_PATH" ] || [ "${TRAIN_ALWAYS}" = "1" ]; then
  echo "Starting initial training..."
  python train.py
else
  echo "Existing model found: $MODEL_PATH"
fi

# Ensure EAs are copied into terminal directory
python scripts/setup_terminal.py /opt/mt5

# Launch MetaTrader under Xvfb if installed
if [ -f /opt/mt5/terminal64.exe ]; then
  echo "Starting MetaTrader..."
  xvfb-run -d wine /opt/mt5/terminal64.exe >/dev/null 2>&1 || true
fi

# Push logs every hour in background
python scripts/hourly_log_push.py &

# Run realtime training loop
exec python realtime_train.py
