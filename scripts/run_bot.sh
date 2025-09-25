#!/usr/bin/env bash
set -e

# Root of repo
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
cd "$REPO_ROOT"

MODEL_PATH="model.joblib"
export CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config.yaml}"
mkdir -p logs

if [ "${START_DASHBOARD:-1}" = "1" ]; then
  DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
  if ! pgrep -f "streamlit run webui/dashboard.py" >/dev/null 2>&1; then
    echo "Launching dashboard on port ${DASHBOARD_PORT}..."
    streamlit run webui/dashboard.py \
      --server.port "${DASHBOARD_PORT}" \
      --server.address "0.0.0.0" \
      >"logs/dashboard.log" 2>&1 &
  else
    echo "Dashboard already running"
  fi
fi

# train if model missing or TRAIN_ALWAYS=1
if [ ! -f "$MODEL_PATH" ] || [ "${TRAIN_ALWAYS}" = "1" ]; then
  echo "Starting initial training..."
  python -m mt5.train
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
python scripts/hourly_artifact_push.py &

# Run realtime training loop
exec python -m mt5.realtime_train
