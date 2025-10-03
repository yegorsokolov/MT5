#!/usr/bin/env bash
set -euo pipefail

run_probe() {
  local probe_output
  local status

  echo "Running MT5 programmatic bridge probe..."
  # Placeholder command simulating the actual probe.
  probe_output=$(cat <<'JSON'
{"status": "ok", "message": "Bridge probe completed"}
JSON
  )
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "Programmatic bridge probe failed with status ${status}." >&2
    return "${status}"
  fi

  if command -v jq >/dev/null 2>&1; then
    if ! jq empty >/dev/null 2>&1 <<<"${probe_output}"; then
      echo "Probe output is not valid JSON." >&2
      return 1
    fi
  else
    if ! python3 -c 'import json,sys; json.load(sys.stdin)' <<<"${probe_output}"; then
      echo "Probe output is not valid JSON." >&2
      return 1
    fi
  fi

  echo "Probe succeeded with response: ${probe_output}"
}

main() {
  run_probe
}

main "$@"
