#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME=mt5bot
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPDATE_SERVICE_NAME=mt5bot-update
UPDATE_SERVICE_FILE="/etc/systemd/system/${UPDATE_SERVICE_NAME}.service"
UPDATE_TIMER_FILE="/etc/systemd/system/${UPDATE_SERVICE_NAME}.timer"

# Ensure runtime controller/config secrets exist for the services.
if [[ "${RUNTIME_SECRETS_SKIP:-0}" == "1" ]]; then
    echo "Skipping runtime secret generation (RUNTIME_SECRETS_SKIP=1)"
else
    SECRET_ENV_FILE="${RUNTIME_SECRETS_ENV_FILE:-${REPO_DIR}/deploy/secrets/runtime.env}"
    SECRET_ARGS=(--env-file "${SECRET_ENV_FILE}")

    if [[ "${RUNTIME_SECRETS_FORCE:-0}" == "1" ]]; then
        SECRET_ARGS+=(--force)
    fi
    if [[ "${RUNTIME_SECRETS_SKIP_CONFIG:-0}" == "1" ]]; then
        SECRET_ARGS+=(--skip-config)
    fi
    if [[ "${RUNTIME_SECRETS_SKIP_CONTROLLER:-0}" == "1" ]]; then
        SECRET_ARGS+=(--skip-controller)
    fi
    if [[ "${RUNTIME_SECRETS_SKIP_ENCRYPTION:-0}" == "1" ]]; then
        SECRET_ARGS+=(--skip-encryption)
    fi
    if [[ -n "${RUNTIME_SECRETS_ROTATE:-}" ]]; then
        for key in ${RUNTIME_SECRETS_ROTATE}; do
            SECRET_ARGS+=(--rotate "${key}")
        done
    fi
    if [[ "${RUNTIME_SECRETS_PRINT_EXPORTS:-0}" == "1" ]]; then
        SECRET_ARGS+=(--print-exports)
    fi

    echo "Ensuring runtime secrets exist (${SECRET_ENV_FILE})"
    (
        cd "${REPO_DIR}"
        python3 -m deployment.runtime_secrets "${SECRET_ARGS[@]}"
    )
fi

# Optionally bootstrap InfluxDB secrets as part of installation.
if [[ -n "${INFLUXDB_BOOTSTRAP_URL:-}" ]]; then
    echo "Bootstrapping InfluxDB metrics bucket"
    : "${INFLUXDB_BOOTSTRAP_ORG:?Set INFLUXDB_BOOTSTRAP_ORG when INFLUXDB_BOOTSTRAP_URL is provided}"
    : "${INFLUXDB_BOOTSTRAP_BUCKET:?Set INFLUXDB_BOOTSTRAP_BUCKET when INFLUXDB_BOOTSTRAP_URL is provided}"

    BOOTSTRAP_ENV_FILE="${INFLUXDB_BOOTSTRAP_ENV_FILE:-${REPO_DIR}/deploy/secrets/influx.env}"
    BOOTSTRAP_ARGS=(
        --url "${INFLUXDB_BOOTSTRAP_URL}"
        --org "${INFLUXDB_BOOTSTRAP_ORG}"
        --bucket "${INFLUXDB_BOOTSTRAP_BUCKET}"
        --env-file "${BOOTSTRAP_ENV_FILE}"
    )

    if [[ -n "${INFLUXDB_BOOTSTRAP_USERNAME:-}" ]]; then
        BOOTSTRAP_ARGS+=(--username "${INFLUXDB_BOOTSTRAP_USERNAME}")
    fi
    if [[ -n "${INFLUXDB_BOOTSTRAP_PASSWORD:-}" ]]; then
        BOOTSTRAP_ARGS+=(--password "${INFLUXDB_BOOTSTRAP_PASSWORD}")
    fi
    if [[ -n "${INFLUXDB_BOOTSTRAP_RETENTION:-}" ]]; then
        BOOTSTRAP_ARGS+=(--retention "${INFLUXDB_BOOTSTRAP_RETENTION}")
    fi
    if [[ -n "${INFLUXDB_BOOTSTRAP_TOKEN_DESCRIPTION:-}" ]]; then
        BOOTSTRAP_ARGS+=(--token-description "${INFLUXDB_BOOTSTRAP_TOKEN_DESCRIPTION}")
    fi
    if [[ -n "${INFLUXDB_BOOTSTRAP_ADMIN_TOKEN:-}" ]]; then
        BOOTSTRAP_ARGS+=(--admin-token "${INFLUXDB_BOOTSTRAP_ADMIN_TOKEN}")
    fi
    if [[ "${INFLUXDB_BOOTSTRAP_STORE_ADMIN:-0}" == "1" ]]; then
        BOOTSTRAP_ARGS+=(--store-admin-secret)
    fi
    if [[ "${INFLUXDB_BOOTSTRAP_FORCE:-0}" == "1" ]]; then
        BOOTSTRAP_ARGS+=(--force)
    fi
    if [[ "${INFLUXDB_BOOTSTRAP_ROTATE_TOKEN:-0}" == "1" ]]; then
        BOOTSTRAP_ARGS+=(--rotate-token)
    fi

    (
        cd "${REPO_DIR}"
        python3 -m deployment.influx_bootstrap "${BOOTSTRAP_ARGS[@]}"
    )
else
    echo "Skipping InfluxDB bootstrap (INFLUXDB_BOOTSTRAP_URL not set)"
fi

# Substitute repository path into unit file and install
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${SERVICE_NAME}.service" | sudo tee "${SERVICE_FILE}" > /dev/null
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${UPDATE_SERVICE_NAME}.service" | sudo tee "${UPDATE_SERVICE_FILE}" > /dev/null
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${UPDATE_SERVICE_NAME}.timer" | sudo tee "${UPDATE_TIMER_FILE}" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl enable --now "${UPDATE_SERVICE_NAME}.timer"
