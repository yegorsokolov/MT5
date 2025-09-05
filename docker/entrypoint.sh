#!/usr/bin/env bash
set -e

python - <<'PY'
from utils.resource_monitor import monitor
from plugins import PLUGIN_SPECS
for spec in PLUGIN_SPECS:
    spec.load()
print(f"Resource tier detected: {monitor.capability_tier}")
PY

exec "$@"
