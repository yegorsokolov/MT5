#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

trim_whitespace() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "${value}"
}

load_project_env() {
    local env_file="$1"
    local line key value

    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%%$'\r'}"
        line="$(trim_whitespace "${line}")"
        if [[ -z "${line}" || "${line}" == \#* ]]; then
            continue
        fi
        if [[ "${line}" =~ ^export[[:space:]]+ ]]; then
            line="${line#export}"
            line="$(trim_whitespace "${line}")"
        fi
        if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            value="$(trim_whitespace "${value}")"
            if [[ ${#value} -ge 2 && "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
                value="${value:1:-1}"
            elif [[ ${#value} -ge 2 && "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
                value="${value:1:-1}"
            fi
            export "${key}=${value}"
        else
            echo "Skipping invalid line in ${env_file}: ${line}" >&2
        fi
    done <"${env_file}"
}

PROJECT_ENV_FILE=""
if env_path="$(${SCRIPT_DIR}/detect_env_file.sh "${PROJECT_ROOT}")"; then
    PROJECT_ENV_FILE="${env_path}"
    export PROJECT_ENV_FILE
    if [[ -f "${PROJECT_ENV_FILE}" ]]; then
        echo "Loading environment overrides from ${PROJECT_ENV_FILE}"
        load_project_env "${PROJECT_ENV_FILE}"
    fi
else
    echo "Warning: Unable to locate a .env file; continuing without local overrides." >&2
fi

SERVICES_ONLY=0
SKIP_SERVICE_INSTALL=0

MT5_WINE_PREFIX="${MT5_WINE_PREFIX:-${WINEPREFIX:-$HOME/.wine-mt5}}"
WIN_PY_WINE_PREFIX="${WIN_PY_WINE_PREFIX:-$HOME/.wine-py311}"
WIN_PY_VERSION="${WIN_PY_VERSION:-3.11.9}"
WIN_PY_INSTALLER="python-${WIN_PY_VERSION}-amd64.exe"
WIN_PY_MAJOR_MINOR="$(echo "${WIN_PY_VERSION}" | awk -F. '{printf "%d%d", $1, $2}')"
WIN_PY_DIR_NAME="Python${WIN_PY_MAJOR_MINOR}"
WIN_PY_UNIX_PATH=""
WIN_PY_WINDOWS_PATH=""
MT5_INSTALL_CACHE="${PROJECT_ROOT:-$(pwd)}/.cache/mt5"
MT5_CACHE_INSTALLER="${MT5_INSTALL_CACHE}/mt5setup.exe"
PYMT5LINUX_SOURCE="${PYMT5LINUX_SOURCE:-pymt5linux}"

ensure_wine_prefix() {
    local prefix="$1"
    echo "Initialising Wine prefix at ${prefix} ..."
    mkdir -p "${prefix}" >/dev/null 2>&1 || true
    WINEPREFIX="${prefix}" WINEARCH=win64 wineboot -u >/dev/null 2>&1 || true
    if command -v wineserver >/dev/null 2>&1; then
        WINEPREFIX="${prefix}" wineserver -w >/dev/null 2>&1 || true
    fi
}

resolve_windows_python_paths() {
    local prefix="$1"
    local candidate
    candidate=$(find "${prefix}/drive_c" -maxdepth 6 -type f -name python.exe 2>/dev/null | \
        grep -E "/${WIN_PY_DIR_NAME}/python.exe$" | head -n 1)
    if [[ -z "${candidate}" ]]; then
        candidate=$(find "${prefix}/drive_c" -maxdepth 6 -type f -name python.exe 2>/dev/null | \
            grep -E '/Python311/python.exe$' | head -n 1)
    fi
    if [[ -n "${candidate}" ]]; then
        WIN_PY_UNIX_PATH="${candidate}"
        WIN_PY_WINDOWS_PATH="$(winepath -w "${candidate}")"
        return 0
    fi
    return 1
}

install_windows_python() {
    local prefix="$1"
    resolve_windows_python_paths "${prefix}" && return 0

    mkdir -p "${MT5_INSTALL_CACHE}" >/dev/null 2>&1 || true
    local installer_path="${MT5_INSTALL_CACHE}/${WIN_PY_INSTALLER}"
    if [[ ! -f "${installer_path}" ]]; then
        echo "Downloading Windows Python ${WIN_PY_VERSION} ..."
        if ! wget -O "${installer_path}" "https://www.python.org/ftp/python/${WIN_PY_VERSION}/${WIN_PY_INSTALLER}"; then
            echo "Warning: Failed to download Windows Python ${WIN_PY_VERSION}." >&2
            rm -f "${installer_path}"
            return 1
        fi
    fi

    echo "Installing Windows Python ${WIN_PY_VERSION} inside Wine prefix ${prefix} ..."
    if ! WINEPREFIX="${prefix}" wine "${installer_path}" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1; then
        echo "Warning: Windows Python installer exited with an error." >&2
        return 1
    fi

    resolve_windows_python_paths "${prefix}"
}

ensure_windows_python_ready() {
    local prefix="${WIN_PY_WINE_PREFIX}"
    ensure_wine_prefix "${prefix}"
    ensure_winetricks_components "${prefix}"
    install_windows_python "${prefix}" || return 1

    if [[ -n "${WIN_PY_UNIX_PATH}" ]]; then
        echo "Upgrading pip inside the Wine Python environment ..."
        WINEPREFIX="${prefix}" wine "${WIN_PY_WINDOWS_PATH}" -m pip install --upgrade pip >/dev/null 2>&1 || true
        return 0
    fi

    echo "Warning: Windows Python executable not found after installation." >&2
    return 1
}

ensure_winetricks_components() {
    local prefix="$1"
    if ! command -v winetricks >/dev/null 2>&1; then
        return 0
    fi
    echo "Installing core Wine runtime components into ${prefix} ..."
    WINEPREFIX="${prefix}" WINEARCH=win64 winetricks -q corefonts gdiplus msxml6 vcrun2019 >/dev/null 2>&1 || true
}

install_mt5_terminal_wine() {
    local prefix="${MT5_WINE_PREFIX}"
    local mt5_dir="${prefix}/drive_c/Program Files/MetaTrader 5"
    local terminal="${mt5_dir}/terminal64.exe"

    ensure_wine_prefix "${prefix}"
    ensure_winetricks_components "${prefix}"
    mkdir -p "${MT5_INSTALL_CACHE}" >/dev/null 2>&1 || true

    if [[ -f "${terminal}" ]]; then
        echo "MetaTrader 5 already installed in Wine prefix (${mt5_dir})."
        return 0
    fi

    local installer_path="${MT5_CACHE_INSTALLER}"
    if [[ ! -f "${installer_path}" ]]; then
        echo "Downloading MetaTrader 5 setup ..."
        if ! wget -O "${installer_path}" "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"; then
            echo "Warning: Failed to download MetaTrader 5 installer." >&2
            rm -f "${installer_path}"
            return 1
        fi
    fi

    echo "Launching MetaTrader 5 installer under Wine. Complete the GUI setup to finish installation."
    if ! WINEPREFIX="${prefix}" wine "${installer_path}" >/dev/null 2>&1; then
        echo "Warning: MetaTrader 5 installer did not finish successfully." >&2
        return 1
    fi
    return 0
}

ensure_mt5_python_packages() {
    if [[ -z "${WIN_PY_UNIX_PATH}" || ! -f "${WIN_PY_UNIX_PATH}" ]]; then
        echo "Warning: Windows Python is not available; skipping MetaTrader5 pip installation." >&2
        return 1
    fi

    echo "Installing MetaTrader5 and bridge helper (${PYMT5LINUX_SOURCE}) inside the Wine Python environment ..."
    if ! WINEPREFIX="${WIN_PY_WINE_PREFIX}" wine "${WIN_PY_WINDOWS_PATH}" -m pip install --upgrade MetaTrader5 "${PYMT5LINUX_SOURCE}" >/dev/null 2>&1; then
        echo "Warning: Failed to install MetaTrader5 or bridge helper inside Wine. Check the Wine logs for more details." >&2
        if [[ "${PYMT5LINUX_SOURCE}" == "pymt5linux" ]]; then
            echo "Hint: export PYMT5LINUX_SOURCE to an alternate package URL if the default is unavailable." >&2
        fi
        return 1
    fi
    return 0
}

write_wine_login_instructions() {
    local instructions_path="${PROJECT_ROOT}/LOGIN_INSTRUCTIONS_WINE.txt"
    local mt5_prefix="${MT5_WINE_PREFIX}"
    local mt5_terminal="${mt5_prefix}/drive_c/Program Files/MetaTrader 5/terminal64.exe"
    if [[ -z "${WIN_PY_WINDOWS_PATH}" ]]; then
        resolve_windows_python_paths "${WIN_PY_WINE_PREFIX}" || true
    fi
    cat >"${instructions_path}" <<EOF
MetaTrader 5 (Wine) login & bridge instructions
===============================================

Terminal prefix : ${mt5_prefix}
Windows Python  : ${WIN_PY_WINDOWS_PATH:-<not detected>}
Terminal path   : ${mt5_terminal}
Bridge exports  : export PYMT5LINUX_PYTHON="${WIN_PY_WINDOWS_PATH:-<not detected>}"
                  export PYMT5LINUX_WINEPREFIX="${WIN_PY_WINE_PREFIX}"

1. Launch the terminal and log in with your broker account:
   WINEARCH=win64 WINEPREFIX="${mt5_prefix}" wine "${mt5_terminal}"
2. (Optional) Start the MetaTrader 5 installer again if you need to repair the
   installation:
   WINEARCH=win64 WINEPREFIX="${mt5_prefix}" wine "${MT5_CACHE_INSTALLER}"
3. To call the MetaTrader 5 Python API directly from Linux, reuse the Windows
   interpreter inside Wine:
   a. Ensure the Windows Python path above is correct. If not, re-run
      scripts/setup_ubuntu.sh to refresh it.
   b. Start the Python bridge from Linux:
      WINEARCH=win64 WINEPREFIX="${WIN_PY_WINE_PREFIX}" \
        wine "${WIN_PY_WINDOWS_PATH:-C:/Python311/python.exe}" -m MetaTrader5 --version
   c. When using helper wrappers (pymt5linux or custom bridge scripts), point
      them at the Windows interpreter recorded above.

This file is generated by scripts/setup_ubuntu.sh. Re-run the script after
updating the Wine prefix or Windows Python version to refresh the paths.
EOF
    echo "MetaTrader 5 Wine instructions saved to ${instructions_path}"
}

prompt_for_mt5_login() {
    local prefix="$1"
    local primary="$2"
    local fallback="$3"

    if [[ "${SKIP_MT5_LOGIN_PROMPT:-0}" == "1" ]]; then
        return
    fi

    if [[ ! -t 0 ]]; then
        echo "Skipping MetaTrader 5 login prompt because the script is running non-interactively." >&2
        return
    fi

    local launch_target=""
    if [[ -n "${primary}" && -f "${primary}" ]]; then
        launch_target="${primary}"
    elif [[ -n "${fallback}" && -f "${fallback}" ]]; then
        launch_target="${fallback}"
    fi

    if [[ -z "${launch_target}" ]]; then
        echo "MetaTrader 5 executable not found; skipping automatic login prompt." >&2
        return
    fi

    if ! command -v wine >/dev/null 2>&1; then
        echo "Wine is not available; skipping automatic MetaTrader 5 launch." >&2
        return
    fi

    echo "Launching MetaTrader 5 so you can complete the initial login..."
    WINEARCH=win64 WINEPREFIX="${prefix}" wine "${launch_target}" >/dev/null 2>&1 &
    local wine_pid=$!
    while true; do
        if ! read -r -p "Did you log into MetaTrader 5 successfully? Type 'yes' to continue: " response; then
            echo "Input closed before confirmation; continuing without verification." >&2
            break
        fi
        response="${response,,}"
        if [[ "${response}" == "yes" ]]; then
            break
        fi
        echo "Please complete the login inside the MetaTrader 5 terminal before proceeding."
    done
    if command -v wineserver >/dev/null 2>&1; then
        WINEPREFIX="${prefix}" wineserver -k >/dev/null 2>&1 || true
    fi
    if kill -0 "${wine_pid}" 2>/dev/null; then
        wait "${wine_pid}" || true
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --services-only|--install-services-only)
            SERVICES_ONLY=1
            shift
            ;;
        --skip-service-install)
            SKIP_SERVICE_INSTALL=1
            shift
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: setup_ubuntu.sh [--services-only] [--skip-service-install]

Without flags the script provisions the Python toolchain, installs project
dependencies, prepares the MetaTrader terminal and installs the systemd
services.  --services-only skips the package and dependency steps and reuses
the existing environment to (re)install the services.  --skip-service-install
performs the environment preparation but leaves the systemd units untouched.
USAGE
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

cd "${PROJECT_ROOT}"

if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    # Remove the unsupported deadsnakes PPA if it exists. Some legacy
    # environments may have had the repository enabled which now causes apt
    # update to fail on newer Ubuntu releases.  The project relies on the
    # distribution provided Python packages, so the extra PPA is not needed.
    if ls /etc/apt/sources.list.d/*deadsnakes* >/dev/null 2>&1; then
        sudo rm -f /etc/apt/sources.list.d/*deadsnakes*
    fi

    if ! dpkg --print-foreign-architectures | grep -q '^i386$'; then
        sudo dpkg --add-architecture i386
    fi
    sudo apt-get update
    sudo apt-get install -y software-properties-common

    echo "Ensuring a supported Python toolchain is available..."
    if ! sudo apt-get install -y python3 python3-venv python3-dev; then
        echo "Warning: Failed to install python3 toolchain packages via apt; attempting to use any existing interpreter." >&2
    fi
    if apt-cache show python3-distutils >/dev/null 2>&1; then
        if ! sudo apt-get install -y python3-distutils; then
            echo "Warning: python3-distutils could not be installed; continuing without it." >&2
        fi
    fi

    sudo apt-get install -y build-essential cabextract wine64 wine32:i386 winetricks
    sudo apt-get install -y wine-gecko2.47.4 wine-gecko2.47.4:i386 wine-mono || true
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" && "${SERVICES_ONLY}" -ne 1 ]]; then
    for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "A supported python3 interpreter was not found after installation attempts." >&2
    exit 1
fi

PYTHON_MAJOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.minor)')
if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 10) )); then
    echo "Python ${PYTHON_MAJOR}.${PYTHON_MINOR} is not supported. Please use Python 3.10 or newer." >&2
    exit 1
fi

export PYTHON_BIN

if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        "$PYTHON_BIN" -m ensurepip --upgrade
    fi

    if ! command -v wget >/dev/null 2>&1; then
        sudo apt-get install -y wget
    fi
fi

if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    echo "Ensuring Linux-side bridge helper (${PYMT5LINUX_SOURCE}) is installed ..."
    if ! "$PYTHON_BIN" -m pip install --upgrade "${PYMT5LINUX_SOURCE}" >/dev/null 2>&1; then
        echo "Warning: Failed to install ${PYMT5LINUX_SOURCE} in the Linux environment." >&2
        if [[ "${PYMT5LINUX_SOURCE}" == "pymt5linux" ]]; then
            echo "Hint: export PYMT5LINUX_SOURCE to point at a Git repository or wheel URL." >&2
        fi
    fi
fi

if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    MT5_DEFAULT_DIR="${MT5_WINE_PREFIX}/drive_c/Program Files/MetaTrader 5"
    MT5_INSTALL_DIR="${MT5_INSTALL_DIR:-${MT5_DEFAULT_DIR}}"
    if [[ "${MT5_INSTALL_DIR}" == "${MT5_DEFAULT_DIR}" ]]; then
        ensure_windows_python_ready || true
        install_mt5_terminal_wine || true
        ensure_mt5_python_packages || true
        write_wine_login_instructions
        mt5_terminal_path="${MT5_DEFAULT_DIR}/terminal64.exe"
        prompt_for_mt5_login "${MT5_WINE_PREFIX}" "${mt5_terminal_path}" "${MT5_CACHE_INSTALLER}"
    else
        MT5_DOWNLOAD_URL="${MT5_DOWNLOAD_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"
        MT5_SETUP_PATH="$MT5_INSTALL_DIR/mt5setup.exe"

        echo "Preparing MetaTrader 5 terminal under $MT5_INSTALL_DIR ..."
        if [ ! -f "$MT5_INSTALL_DIR/terminal64.exe" ] && [ ! -f "$MT5_SETUP_PATH" ]; then
            if [[ "$MT5_INSTALL_DIR" == /opt/* ]]; then
                sudo mkdir -p "$MT5_INSTALL_DIR"
            else
                mkdir -p "$MT5_INSTALL_DIR"
            fi
            tmpfile="$(mktemp)"
            echo "Downloading MetaTrader 5 setup from $MT5_DOWNLOAD_URL"
            if wget -O "$tmpfile" "$MT5_DOWNLOAD_URL"; then
                if [[ "$MT5_INSTALL_DIR" == /opt/* ]]; then
                    sudo mv "$tmpfile" "$MT5_SETUP_PATH"
                    sudo chown root:root "$MT5_SETUP_PATH"
                    sudo chmod 755 "$MT5_SETUP_PATH"
                else
                    mv "$tmpfile" "$MT5_SETUP_PATH"
                    chmod 755 "$MT5_SETUP_PATH"
                fi
                cat >"$MT5_INSTALL_DIR/LOGIN_INSTRUCTIONS.txt" <<EOF
MetaTrader 5 login instructions
================================

1. Launch the installer once: `wine "$MT5_SETUP_PATH"`.
2. Complete the platform installation when prompted.
3. Sign in with your broker credentials so historical data can be downloaded.
4. Close the terminal once login succeeds. The training pipeline will reuse
   the authenticated terminal to synchronise price history before training.

If you reinstall MetaTrader 5 elsewhere update MT5_INSTALL_DIR before running
setup_ubuntu.sh.
EOF
                echo "MetaTrader 5 setup downloaded to $MT5_SETUP_PATH."
            else
                echo "Warning: Failed to download MetaTrader 5 setup." >&2
                rm -f "$tmpfile"
            fi
        else
            echo "MetaTrader 5 already installed at $MT5_INSTALL_DIR"
        fi

        prompt_for_mt5_login "${MT5_WINE_PREFIX}" "$MT5_INSTALL_DIR/terminal64.exe" "$MT5_SETUP_PATH"
    fi

    # Optionally install CUDA drivers if an NVIDIA GPU is detected or WITH_CUDA=1
    if command -v nvidia-smi >/dev/null 2>&1 || [[ "${WITH_CUDA:-0}" == "1" ]]; then
        sudo apt-get install -y nvidia-cuda-toolkit
    fi
fi

provision_mt5bot_service() {
    local service_name="mt5bot"
    local service_file="/etc/systemd/system/${service_name}.service"
    local update_service_name="mt5bot-update"
    local update_service_file="/etc/systemd/system/${update_service_name}.service"
    local update_timer_file="/etc/systemd/system/${update_service_name}.timer"

    local env_file="${PROJECT_ENV_FILE:-${PROJECT_ROOT}/.env}"
    if [[ -n "${env_file}" && -f "${env_file}" ]]; then
        echo "Using environment file ${env_file}"
    else
        echo "Warning: Environment file not found; expected at ${env_file}." >&2
    fi

    echo "Detecting MetaTrader 5 terminal path for service configuration ..."
    if ! "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/detect_mt5_terminal.py"; then
        echo "Warning: Unable to determine the MetaTrader 5 terminal path automatically." >&2
    fi

    if [[ "${INSTALL_SKIP_ENV_CHECK:-0}" == "1" ]]; then
        echo "Skipping environment diagnostics (INSTALL_SKIP_ENV_CHECK=1)"
    else
        echo "Running environment diagnostics (utils.environment)"
        if ! (
            cd "${PROJECT_ROOT}"
            "${PYTHON_BIN}" -m utils.environment --no-auto-install --strict
        ); then
            echo "Environment diagnostics failed; resolve the issues above and rerun the installer." >&2
            exit 1
        fi
    fi

    echo "Verifying MetaTrader 5 connectivity (reusing any running terminal session) ..."
    if ! "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/setup_terminal.py" --install-heartbeat; then
        echo "Warning: MetaTrader 5 connectivity check failed. Continue after ensuring the terminal is running and logged in." >&2
    fi

    if [[ "${RUNTIME_SECRETS_SKIP:-0}" == "1" ]]; then
        echo "Skipping runtime secret generation (RUNTIME_SECRETS_SKIP=1)"
    else
        local secret_env_file="${RUNTIME_SECRETS_ENV_FILE:-${PROJECT_ROOT}/deploy/secrets/runtime.env}"
        local secret_args=(--env-file "${secret_env_file}")

        if [[ "${RUNTIME_SECRETS_FORCE:-0}" == "1" ]]; then
            secret_args+=(--force)
        fi
        if [[ "${RUNTIME_SECRETS_SKIP_CONFIG:-0}" == "1" ]]; then
            secret_args+=(--skip-config)
        fi
        if [[ "${RUNTIME_SECRETS_SKIP_CONTROLLER:-0}" == "1" ]]; then
            secret_args+=(--skip-controller)
        fi
        if [[ "${RUNTIME_SECRETS_SKIP_ENCRYPTION:-0}" == "1" ]]; then
            secret_args+=(--skip-encryption)
        fi
        if [[ -n "${RUNTIME_SECRETS_ROTATE:-}" ]]; then
            for key in ${RUNTIME_SECRETS_ROTATE}; do
                secret_args+=(--rotate "${key}")
            done
        fi
        if [[ "${RUNTIME_SECRETS_PRINT_EXPORTS:-0}" == "1" ]]; then
            secret_args+=(--print-exports)
        fi

        echo "Ensuring runtime secrets exist (${secret_env_file})"
        (
            cd "${PROJECT_ROOT}"
            "${PYTHON_BIN}" -m deployment.runtime_secrets "${secret_args[@]}"
        )
    fi

    if [[ "${PROM_URLS_SKIP:-0}" == "1" ]]; then
        echo "Skipping Prometheus URL generation (PROM_URLS_SKIP=1)"
    else
        local prom_env_file="${PROM_URLS_ENV_FILE:-${PROJECT_ROOT}/deploy/secrets/runtime.env}"
        local prom_args=(--env-file "${prom_env_file}")

        if [[ "${PROM_URLS_FORCE:-0}" == "1" ]]; then
            prom_args+=(--force)
        fi
        if [[ "${PROM_URLS_DISABLE_PUSH:-0}" == "1" ]]; then
            prom_args+=(--disable-push)
        fi
        if [[ "${PROM_URLS_DISABLE_QUERY:-0}" == "1" ]]; then
            prom_args+=(--disable-query)
        fi

        if [[ -n "${PROM_URLS_PUSH_URL:-}" ]]; then
            prom_args+=(--push-url "${PROM_URLS_PUSH_URL}")
        else
            if [[ -n "${PROM_URLS_PUSH_SCHEME:-}" ]]; then
                prom_args+=(--push-scheme "${PROM_URLS_PUSH_SCHEME}")
            fi
            if [[ -n "${PROM_URLS_PUSH_HOST:-}" ]]; then
                prom_args+=(--push-host "${PROM_URLS_PUSH_HOST}")
            fi
            if [[ -n "${PROM_URLS_PUSH_PORT:-}" ]]; then
                prom_args+=(--push-port "${PROM_URLS_PUSH_PORT}")
            fi
            if [[ -n "${PROM_URLS_PUSH_PATH:-}" ]]; then
                prom_args+=(--push-path "${PROM_URLS_PUSH_PATH}")
            fi
            if [[ -n "${PROM_URLS_PUSH_JOB:-}" ]]; then
                prom_args+=(--push-job "${PROM_URLS_PUSH_JOB}")
            fi
            if [[ -n "${PROM_URLS_PUSH_INSTANCE:-}" ]]; then
                prom_args+=(--push-instance "${PROM_URLS_PUSH_INSTANCE}")
            fi
        fi

        if [[ -n "${PROM_URLS_QUERY_URL:-}" ]]; then
            prom_args+=(--query-url "${PROM_URLS_QUERY_URL}")
        else
            if [[ -n "${PROM_URLS_QUERY_SCHEME:-}" ]]; then
                prom_args+=(--query-scheme "${PROM_URLS_QUERY_SCHEME}")
            fi
            if [[ -n "${PROM_URLS_QUERY_HOST:-}" ]]; then
                prom_args+=(--query-host "${PROM_URLS_QUERY_HOST}")
            fi
            if [[ -n "${PROM_URLS_QUERY_PORT:-}" ]]; then
                prom_args+=(--query-port "${PROM_URLS_QUERY_PORT}")
            fi
            if [[ -n "${PROM_URLS_QUERY_PATH:-}" ]]; then
                prom_args+=(--query-path "${PROM_URLS_QUERY_PATH}")
            fi
        fi

        if [[ "${PROM_URLS_PRINT_EXPORTS:-0}" == "1" ]]; then
            prom_args+=(--print-exports)
        fi

        echo "Ensuring Prometheus URLs exist (${prom_env_file})"
        (
            cd "${PROJECT_ROOT}"
            "${PYTHON_BIN}" -m deployment.prometheus_urls "${prom_args[@]}"
        )
    fi

    if [[ -n "${INFLUXDB_BOOTSTRAP_URL:-}" ]]; then
        echo "Bootstrapping InfluxDB metrics bucket"
        : "${INFLUXDB_BOOTSTRAP_ORG:?Set INFLUXDB_BOOTSTRAP_ORG when INFLUXDB_BOOTSTRAP_URL is provided}"
        : "${INFLUXDB_BOOTSTRAP_BUCKET:?Set INFLUXDB_BOOTSTRAP_BUCKET when INFLUXDB_BOOTSTRAP_URL is provided}"

        local bootstrap_env_file="${INFLUXDB_BOOTSTRAP_ENV_FILE:-${PROJECT_ROOT}/deploy/secrets/influx.env}"
        local bootstrap_args=(
            --url "${INFLUXDB_BOOTSTRAP_URL}"
            --org "${INFLUXDB_BOOTSTRAP_ORG}"
            --bucket "${INFLUXDB_BOOTSTRAP_BUCKET}"
            --env-file "${bootstrap_env_file}"
        )

        if [[ -n "${INFLUXDB_BOOTSTRAP_USERNAME:-}" ]]; then
            bootstrap_args+=(--username "${INFLUXDB_BOOTSTRAP_USERNAME}")
        fi
        if [[ -n "${INFLUXDB_BOOTSTRAP_PASSWORD:-}" ]]; then
            bootstrap_args+=(--password "${INFLUXDB_BOOTSTRAP_PASSWORD}")
        fi
        if [[ -n "${INFLUXDB_BOOTSTRAP_RETENTION:-}" ]]; then
            bootstrap_args+=(--retention "${INFLUXDB_BOOTSTRAP_RETENTION}")
        fi
        if [[ -n "${INFLUXDB_BOOTSTRAP_TOKEN_DESCRIPTION:-}" ]]; then
            bootstrap_args+=(--token-description "${INFLUXDB_BOOTSTRAP_TOKEN_DESCRIPTION}")
        fi
        if [[ -n "${INFLUXDB_BOOTSTRAP_ADMIN_TOKEN:-}" ]]; then
            bootstrap_args+=(--admin-token "${INFLUXDB_BOOTSTRAP_ADMIN_TOKEN}")
        fi
        if [[ "${INFLUXDB_BOOTSTRAP_STORE_ADMIN:-0}" == "1" ]]; then
            bootstrap_args+=(--store-admin-secret)
        fi
        if [[ "${INFLUXDB_BOOTSTRAP_FORCE:-0}" == "1" ]]; then
            bootstrap_args+=(--force)
        fi
        if [[ "${INFLUXDB_BOOTSTRAP_ROTATE_TOKEN:-0}" == "1" ]]; then
            bootstrap_args+=(--rotate-token)
        fi

        (
            cd "${PROJECT_ROOT}"
            "${PYTHON_BIN}" -m deployment.influx_bootstrap "${bootstrap_args[@]}"
        )
    else
        echo "Skipping InfluxDB bootstrap (INFLUXDB_BOOTSTRAP_URL not set)"
    fi

    echo "Installing systemd units..."
    sudo sed "s|{{REPO_PATH}}|${PROJECT_ROOT}|g" "deploy/${service_name}.service" | sudo tee "${service_file}" >/dev/null
    sudo sed "s|{{REPO_PATH}}|${PROJECT_ROOT}|g" "deploy/${update_service_name}.service" | sudo tee "${update_service_file}" >/dev/null
    sudo sed "s|{{REPO_PATH}}|${PROJECT_ROOT}|g" "deploy/${update_service_name}.timer" | sudo tee "${update_timer_file}" >/dev/null

    sudo systemctl daemon-reload
    sudo systemctl enable --now "${service_name}"
    sudo systemctl enable --now "${update_service_name}.timer"
    sudo systemctl status --no-pager "${service_name}" || true
    sudo systemctl status --no-pager "${update_service_name}.timer" || true
}

if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    PIP_CMD=("$PYTHON_BIN" -m pip)

    echo "Checking for outdated Python packages before installation..."
    "${PIP_CMD[@]}" list --outdated || true

    echo "Upgrading pip to the latest version..."
    "${PIP_CMD[@]}" install --upgrade pip

    if "${PIP_CMD[@]}" show ydata-synthetic >/dev/null 2>&1; then
        echo "Removing legacy ydata-synthetic package that is incompatible with modern Python versions..."
        "${PIP_CMD[@]}" uninstall -y ydata-synthetic
    fi

    echo "Installing the latest compatible versions of project dependencies..."
    "${PIP_CMD[@]}" install --upgrade --upgrade-strategy eager -r requirements.txt

    echo "Running project package synchronisation script..."
    ./scripts/update_python_packages.sh

    if (( PYTHON_MAJOR == 3 && PYTHON_MINOR <= 12 )); then
        missing_packages=()
        if ! "$PYTHON_BIN" -c "import ray" >/dev/null 2>&1; then
            missing_packages+=("ray[default]")
        fi
        if ! "$PYTHON_BIN" -c "import torch_geometric" >/dev/null 2>&1; then
            missing_packages+=("torch-geometric")
        fi
        if ! "$PYTHON_BIN" -c "import uvloop" >/dev/null 2>&1; then
            missing_packages+=("uvloop")
        fi

        if (( ${#missing_packages[@]} )); then
            echo "Installing distributed training dependencies: ${missing_packages[*]}"
            if ! "${PIP_CMD[@]}" install --upgrade "${missing_packages[@]}"; then
                echo "Warning: Failed to install distributed training dependencies (${missing_packages[*]})." >&2
                echo "Distributed and graph trainers will fall back to pure-Python implementations." >&2
            fi
        else
            echo "Ray, torch-geometric and uvloop already available; distributed trainers enabled."
        fi
    else
        echo "Skipping Ray/torch-geometric installation on Python ${PYTHON_MAJOR}.${PYTHON_MINOR}."
    fi

    echo "Outdated packages remaining after upgrade (if any):"
    "${PIP_CMD[@]}" list --outdated || true
fi

if [[ "${SKIP_SERVICE_INSTALL}" -ne 1 ]]; then
    echo "Installing and starting the MT5 bot service..."
    provision_mt5bot_service
else
    echo "Skipping service installation (--skip-service-install)."
fi

echo "Triggering an immediate MT5 bot update check..."
"$PYTHON_BIN" -m services.auto_updater --force

if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    echo "Recording environment diagnostics for reproducibility..."
    "$PYTHON_BIN" -m utils.environment --json || true

    echo "AutoGluon has been replaced with the built-in tabular trainer."
    echo "Run 'python -m mt5.train_tabular' after setup to train the default model."
fi
