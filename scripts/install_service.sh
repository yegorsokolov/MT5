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

# Generate Prometheus Pushgateway/Prometheus endpoints alongside runtime secrets.
if [[ "${PROM_ENDPOINTS_SKIP:-0}" == "1" ]]; then
    echo "Skipping Prometheus endpoint generation (PROM_ENDPOINTS_SKIP=1)"
else
    PROM_ENV_FILE="${PROM_ENDPOINTS_ENV_FILE:-${REPO_DIR}/deploy/secrets/runtime.env}"
    PROM_ARGS=(--env-file "${PROM_ENV_FILE}")

    if [[ "${PROM_ENDPOINTS_FORCE:-0}" == "1" ]]; then
        PROM_ARGS+=(--force)
    fi
    if [[ "${PROM_ENDPOINTS_SKIP_PUSH:-0}" == "1" ]]; then
        PROM_ARGS+=(--skip-push)
    fi
    if [[ "${PROM_ENDPOINTS_SKIP_QUERY:-0}" == "1" ]]; then
        PROM_ARGS+=(--skip-query)
    fi
    if [[ "${PROM_ENDPOINTS_PRINT_EXPORTS:-0}" == "1" ]]; then
        PROM_ARGS+=(--print-exports)
    fi

    if [[ "${PROM_ENDPOINTS_SKIP_PUSH:-0}" != "1" ]]; then
        for var in \
            PROM_ENDPOINTS_PUSH_URL \
            PROM_ENDPOINTS_PUSH_SCHEME \
            PROM_ENDPOINTS_PUSH_HOST \
            PROM_ENDPOINTS_PUSH_PORT \
            PROM_ENDPOINTS_PUSH_PATH \
            PROM_ENDPOINTS_PUSH_JOB \
            PROM_ENDPOINTS_PUSH_INSTANCE; do
            value="${!var:-}"
            if [[ -z "${value}" ]]; then
                continue
            fi
            case "${var}" in
                PROM_ENDPOINTS_PUSH_URL)
                    PROM_ARGS+=(--push-url "${value}")
                    ;;
                PROM_ENDPOINTS_PUSH_SCHEME)
                    PROM_ARGS+=(--push-scheme "${value}")
                    ;;
                PROM_ENDPOINTS_PUSH_HOST)
                    PROM_ARGS+=(--push-host "${value}")
                    ;;
                PROM_ENDPOINTS_PUSH_PORT)
                    PROM_ARGS+=(--push-port "${value}")
                    ;;
                PROM_ENDPOINTS_PUSH_PATH)
                    PROM_ARGS+=(--push-path "${value}")
                    ;;
                PROM_ENDPOINTS_PUSH_JOB)
                    PROM_ARGS+=(--push-job "${value}")
                    ;;
                PROM_ENDPOINTS_PUSH_INSTANCE)
                    PROM_ARGS+=(--push-instance "${value}")
                    ;;
            esac
        done
    fi

    if [[ "${PROM_ENDPOINTS_SKIP_QUERY:-0}" != "1" ]]; then
        for var in \
            PROM_ENDPOINTS_QUERY_URL \
            PROM_ENDPOINTS_QUERY_SCHEME \
            PROM_ENDPOINTS_QUERY_HOST \
            PROM_ENDPOINTS_QUERY_PORT \
            PROM_ENDPOINTS_QUERY_PATH; do
            value="${!var:-}"
            if [[ -z "${value}" ]]; then
                continue
            fi
            case "${var}" in
                PROM_ENDPOINTS_QUERY_URL)
                    PROM_ARGS+=(--query-url "${value}")
                    ;;
                PROM_ENDPOINTS_QUERY_SCHEME)
                    PROM_ARGS+=(--query-scheme "${value}")
                    ;;
                PROM_ENDPOINTS_QUERY_HOST)
                    PROM_ARGS+=(--query-host "${value}")
                    ;;
                PROM_ENDPOINTS_QUERY_PORT)
                    PROM_ARGS+=(--query-port "${value}")
                    ;;
                PROM_ENDPOINTS_QUERY_PATH)
                    PROM_ARGS+=(--query-path "${value}")
                    ;;
            esac
        done
    fi

    echo "Ensuring Prometheus endpoint URLs exist (${PROM_ENV_FILE})"
    (
        cd "${REPO_DIR}"
        python3 -m deployment.prometheus_endpoints "${PROM_ARGS[@]}"
    )
fi

# Substitute repository path into unit file and install
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${SERVICE_NAME}.service" | sudo tee "${SERVICE_FILE}" > /dev/null
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${UPDATE_SERVICE_NAME}.service" | sudo tee "${UPDATE_SERVICE_FILE}" > /dev/null
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${UPDATE_SERVICE_NAME}.timer" | sudo tee "${UPDATE_TIMER_FILE}" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl enable --now "${UPDATE_SERVICE_NAME}.timer"
