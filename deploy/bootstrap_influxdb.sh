#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INFLUX_BIN=${INFLUX_BIN:-influx}
INFLUX_URL=${INFLUX_URL:-http://localhost:8086}
INFLUX_ORG=${INFLUX_ORG:-mt5}
INFLUX_BUCKET=${INFLUX_BUCKET:-mt5_metrics}
INFLUX_USERNAME=${INFLUX_USERNAME:-mt5-admin}
INFLUX_PASSWORD=${INFLUX_PASSWORD:-}
INFLUX_TOKEN=${INFLUX_TOKEN:-}
SECRETS_DIR=${SECRETS_DIR:-"${REPO_DIR}/deploy/secrets"}
ENV_FILE=${ENV_FILE:-"${SECRETS_DIR}/influx.env"}

random_hex() {
  python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
}

random_password() {
  python - <<'PY'
import secrets
import string
alphabet = string.ascii_letters + string.digits
print(''.join(secrets.choice(alphabet) for _ in range(32)))
PY
}

if [[ -f "${ENV_FILE}" ]]; then
  echo "Found existing InfluxDB credentials at ${ENV_FILE}; skipping bootstrap."
  exit 0
fi

if ! command -v "${INFLUX_BIN}" >/dev/null 2>&1; then
  echo "Error: influx CLI not found on PATH. Install it from https://docs.influxdata.com/influxdb/." >&2
  exit 1
fi

if ! "${INFLUX_BIN}" ping --host "${INFLUX_URL}" >/dev/null 2>&1; then
  echo "Error: unable to reach InfluxDB at ${INFLUX_URL}. Ensure the server is running." >&2
  exit 1
fi

if [[ -z "${INFLUX_PASSWORD}" ]]; then
  INFLUX_PASSWORD=$(random_password)
fi

if [[ -z "${INFLUX_TOKEN}" ]]; then
  INFLUX_TOKEN=$(random_hex)
fi

mkdir -p "${SECRETS_DIR}"

echo "Bootstrapping InfluxDB organisation '${INFLUX_ORG}' and bucket '${INFLUX_BUCKET}'."

set +e
setup_output=$("${INFLUX_BIN}" setup \
  --host "${INFLUX_URL}" \
  --org "${INFLUX_ORG}" \
  --bucket "${INFLUX_BUCKET}" \
  --username "${INFLUX_USERNAME}" \
  --password "${INFLUX_PASSWORD}" \
  --token "${INFLUX_TOKEN}" \
  --retention 0 \
  --force 2>&1)
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  if grep -q "has already been set up" <<<"${setup_output}"; then
    cat >&2 <<'MSG'
Error: InfluxDB has already been initialised. Provide INFLUX_TOKEN/INFLUX_PASSWORD
values via environment variables or delete the existing instance before rerunning.
MSG
  else
    echo "${setup_output}" >&2
  fi
  exit 1
fi

echo "${setup_output}"

cat > "${ENV_FILE}" <<EOF_ENV
INFLUXDB_URL=${INFLUX_URL}
INFLUXDB_TOKEN=${INFLUX_TOKEN}
INFLUXDB_ORG=${INFLUX_ORG}
INFLUXDB_BUCKET=${INFLUX_BUCKET}
EOF_ENV

chmod 600 "${ENV_FILE}"

cat <<INFO
Stored InfluxDB credentials in ${ENV_FILE}.
Admin username: ${INFLUX_USERNAME}
Admin password: ${INFLUX_PASSWORD}
Token: ${INFLUX_TOKEN}
INFO
