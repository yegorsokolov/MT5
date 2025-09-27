#!/usr/bin/env bash
# shellcheck disable=SC2317
set -euo pipefail

# Unified provisioning script that installs Wine + MetaTrader 5, prepares the
# Linux Python environment and verifies the MT5 bridge from the current repo.
#
# The script combines the responsibilities that previously lived in
# deploy_mt5.sh, setup_ubuntu.sh and setup_terminal.py so operators only need a
# single entrypoint.  It is designed to be idempotent and to continue after
# recoverable failures while surfacing actionable diagnostics for non
# recoverable errors.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${PROJECT_ROOT}/.cache/mt5"

MT5_WINE_PREFIX_DEFAULT="$HOME/.wine-mt5"
WIN_PY_WINE_PREFIX_DEFAULT="$HOME/.wine-py311"

MT5_WINE_PREFIX="${MT5_WINE_PREFIX:-${MT5_WINE_PREFIX_DEFAULT}}"
WIN_PY_WINE_PREFIX="${WIN_PY_WINE_PREFIX:-${WIN_PY_WINE_PREFIX_DEFAULT}}"
WIN_PY_VERSION="${WIN_PY_VERSION:-3.11.9}"
WIN_PY_INSTALLER="python-${WIN_PY_VERSION}-amd64.exe"
WIN_PY_URL="https://www.python.org/ftp/python/${WIN_PY_VERSION}/${WIN_PY_INSTALLER}"

MT5_SETUP_URL="${MT5_SETUP_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"
MT5_INSTALLER_PATH="${CACHE_DIR}/mt5setup.exe"

MT5_DIR="${MT5_DIR:-${MT5_WINE_PREFIX}/drive_c/Program Files/MetaTrader 5}"
MT5_TERMINAL="${MT5_TERMINAL:-${MT5_DIR}/terminal64.exe}"

ENV_FILE="${PROJECT_ROOT}/.env"
LOGIN_INSTRUCTIONS="${PROJECT_ROOT}/LOGIN_INSTRUCTIONS_WINE.txt"

LINUX_PYTHON_BIN="${PYTHON_BIN:-}"
CREATE_VENV=1
NON_INTERACTIVE=0
SKIP_LINUX_ENV=0
SKIP_SERVICE_INSTALL=0
SKIP_CONNECTIVITY_CHECK=0

usage() {
    cat <<'USAGE'
Usage: provision_mt5_environment.sh [options]

Options:
  --python-bin PATH          Use a specific python3 interpreter for the Linux environment.
  --reuse-existing-venv      Do not create .venv automatically (assume it's prepared).
  --non-interactive          Skip login prompts and heartbeat installation questions.
  --skip-linux-env           Only prepare the Wine/Windwos side (skip Linux deps install).
  --skip-service-install     Do not install systemd services (mirrors setup_ubuntu.sh option).
  --skip-connectivity-check  Skip the final MetaTrader bridge verification step.
  -h, --help                 Display this help message.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-bin)
            LINUX_PYTHON_BIN="$2"
            shift 2
            ;;
        --reuse-existing-venv)
            CREATE_VENV=0
            shift
            ;;
        --non-interactive)
            NON_INTERACTIVE=1
            shift
            ;;
        --skip-linux-env)
            SKIP_LINUX_ENV=1
            shift
            ;;
        --skip-service-install)
            SKIP_SERVICE_INSTALL=1
            shift
            ;;
        --skip-connectivity-check)
            SKIP_CONNECTIVITY_CHECK=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

log() {
    printf '\e[1;34m>>> %s\e[0m\n' "$*"
}

warn() {
    printf '\e[1;33m*** %s\e[0m\n' "$*" >&2
}

die() {
    printf '\e[1;31m!!! %s\e[0m\n' "$*" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

to_windows_path() {
    local path="$1"
    if command_exists winepath; then
        winepath -w "$path" 2>/dev/null || printf '%s' "$path"
    else
        printf '%s' "$path"
    fi
}

require_command() {
    local cmd="$1"
    command_exists "$cmd" || die "Required command '$cmd' not found in PATH."
}

ensure_directory() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
    fi
}

run_with_sudo() {
    if command_exists sudo; then
        sudo "$@"
    else
        "$@"
    fi
}

ensure_apt_packages() {
    if ! command_exists apt-get; then
        warn "apt-get not available; skipping package installation. Ensure Wine and dependencies are already installed."
        return
    fi

    log "Ensuring multi-arch support for i386 packages"
    if ! dpkg --print-foreign-architectures | grep -q '^i386$'; then
        run_with_sudo dpkg --add-architecture i386 || warn "Unable to add i386 architecture; Wine 32-bit components may be missing."
    fi

    log "Updating apt repositories"
    if ! run_with_sudo apt-get update; then
        warn "apt-get update failed; continuing with cached package metadata."
    fi

    local packages=(software-properties-common build-essential cabextract wget unzip git python3 python3-venv python3-dev wine64 wine32:i386 winetricks)

    log "Installing required apt packages: ${packages[*]}"
    if ! run_with_sudo apt-get install -y "${packages[@]}"; then
        warn "Failed to install some apt packages. Ensure Wine and Python are installed before continuing."
    fi

    if run_with_sudo apt-get install -y wine-gecko2.47.4 wine-gecko2.47.4:i386 wine-mono; then
        log "Installed Wine gecko/mono support packages."
    else
        warn "Unable to install Wine gecko/mono support packages; continuing."
    fi
}

ensure_wine_prefix() {
    local prefix="$1"
    if [[ -z "$prefix" ]]; then
        return
    fi

    log "Initialising Wine prefix at $prefix"
    ensure_directory "$prefix"
    WINEARCH=win64 WINEPREFIX="$prefix" wineboot --init >/dev/null 2>&1 || warn "wineboot initialisation exited with non-zero status."
    if command_exists wineserver; then
        WINEPREFIX="$prefix" wineserver -w >/dev/null 2>&1 || true
    fi
}

install_winetricks_runtime() {
    local prefix="$1"
    if ! command_exists winetricks; then
        warn "winetricks not available; skipping VC++ runtime installation."
        return
    fi

    log "Installing VC++ runtime libraries into $prefix"
    if ! WINEARCH=win64 WINEPREFIX="$prefix" winetricks -q vcrun2022 corefonts gdiplus msxml6 >/dev/null 2>&1; then
        warn "winetricks reported errors installing runtimes. Wine may still work if the components were previously installed."
    fi

    cat >"${CACHE_DIR}/dlloverrides.reg" <<'REGFILE'
REGEDIT4

[HKEY_CURRENT_USER\Software\Wine\DllOverrides]
"ucrtbase"="native,builtin"
"msvcp140"="native,builtin"
"vcruntime140"="native,builtin"
"vcruntime140_1"="native,builtin"
REGFILE
    WINEPREFIX="$prefix" wine regedit /S "${CACHE_DIR}/dlloverrides.reg" >/dev/null 2>&1 || warn "Unable to apply DLL override registry settings."
}

wine_find_python() {
    local prefix="$1"
    find "$prefix/drive_c" -maxdepth 6 -type f -name python.exe 2>/dev/null | grep -Ei 'Python3(1[01]|\d+)/python.exe$' | head -n1
}

download_file() {
    local url="$1"; shift
    local destination="$1"; shift
    local label="$1"

    if [[ -f "$destination" ]]; then
        log "$label already downloaded: $destination"
        return 0
    fi

    ensure_directory "$(dirname "$destination")"
    log "Downloading $label from $url"
    if command_exists curl; then
        if ! curl -fL "$url" -o "$destination"; then
            rm -f "$destination"
            return 1
        fi
    else
        if ! wget -O "$destination" "$url"; then
            rm -f "$destination"
            return 1
        fi
    fi
    return 0
}

install_windows_python() {
    ensure_wine_prefix "$WIN_PY_WINE_PREFIX"
    install_winetricks_runtime "$WIN_PY_WINE_PREFIX"

    local python_exe
    python_exe="$(wine_find_python "$WIN_PY_WINE_PREFIX")"
    if [[ -n "$python_exe" ]]; then
        log "Windows Python already present at $python_exe"
        WIN_PY_UNIX_PATH="$python_exe"
        WIN_PY_WINDOWS_PATH="$(to_windows_path "$python_exe")"
        return 0
    fi

    local installer_path="${CACHE_DIR}/${WIN_PY_INSTALLER}"
    if ! download_file "$WIN_PY_URL" "$installer_path" "Windows Python ${WIN_PY_VERSION}"; then
        warn "Failed to download Windows Python ${WIN_PY_VERSION}."
        return 1
    fi

    log "Installing Windows Python ${WIN_PY_VERSION} inside $WIN_PY_WINE_PREFIX"
    if ! WINEPREFIX="$WIN_PY_WINE_PREFIX" wine "$installer_path" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 >/dev/null 2>&1; then
        warn "Windows Python installer exited with a non-zero status."
    fi

    python_exe="$(wine_find_python "$WIN_PY_WINE_PREFIX")"
    if [[ -z "$python_exe" ]]; then
        warn "Unable to locate python.exe after installation."
        return 1
    fi

    WIN_PY_UNIX_PATH="$python_exe"
    WIN_PY_WINDOWS_PATH="$(to_windows_path "$python_exe")"

    WINEPREFIX="$WIN_PY_WINE_PREFIX" wine "$WIN_PY_WINDOWS_PATH" -m pip install --upgrade pip >/dev/null 2>&1 || warn "Failed to upgrade pip inside Wine Python."
    return 0
}

install_mt5_terminal() {
    ensure_wine_prefix "$MT5_WINE_PREFIX"
    install_winetricks_runtime "$MT5_WINE_PREFIX"

    if [[ -f "$MT5_TERMINAL" ]]; then
        log "MetaTrader 5 terminal already present at $MT5_TERMINAL"
        return 0
    fi

    if ! download_file "$MT5_SETUP_URL" "$MT5_INSTALLER_PATH" "MetaTrader 5"; then
        warn "Failed to download MetaTrader 5 installer."
        return 1
    fi

    log "Launching MetaTrader 5 installer. Complete the GUI setup when prompted."
    if ! WINEPREFIX="$MT5_WINE_PREFIX" wine "$MT5_INSTALLER_PATH" >/dev/null 2>&1; then
        warn "MetaTrader 5 installer exited with a non-zero status."
    fi

    if [[ ! -f "$MT5_TERMINAL" ]]; then
        warn "MetaTrader 5 terminal executable still missing after installation."
        return 1
    fi
    return 0
}

install_mt5_windows_packages() {
    if [[ -z "${WIN_PY_WINDOWS_PATH:-}" ]]; then
        warn "Windows Python path unknown; skipping MetaTrader5 pip installation."
        return
    fi

    log "Installing MetaTrader5 and bridge helpers inside the Wine Python environment"
    local packages=("MetaTrader5")
    if [[ -n "${PYMT5LINUX_SOURCE:-}" ]]; then
        packages+=("${PYMT5LINUX_SOURCE}")
    else
        packages+=("pymt5linux")
    fi

    if ! WINEPREFIX="$WIN_PY_WINE_PREFIX" wine "$WIN_PY_WINDOWS_PATH" -m pip install --upgrade "${packages[@]}"; then
        warn "Failed to install MetaTrader5/pymt5linux inside Wine Python."
    fi
}

ensure_linux_python() {
    if [[ -n "$LINUX_PYTHON_BIN" ]]; then
        require_command "$LINUX_PYTHON_BIN"
        return
    fi

    for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
        if command_exists "$candidate"; then
            LINUX_PYTHON_BIN="$(command -v "$candidate")"
            return
        fi
    done
    die "No supported python3 interpreter found."
}

create_linux_venv() {
    if [[ "$CREATE_VENV" -eq 0 ]]; then
        log "Reusing existing Python environment (requested via --reuse-existing-venv)."
        return
    fi

    ensure_linux_python
    local venv_path="${PROJECT_ROOT}/.venv"
    if [[ -d "$venv_path" ]]; then
        log "Virtual environment already exists at $venv_path"
    else
        log "Creating virtual environment at $venv_path using $LINUX_PYTHON_BIN"
        "$LINUX_PYTHON_BIN" -m venv "$venv_path"
    fi
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
}

install_linux_requirements() {
    if [[ "$SKIP_LINUX_ENV" -eq 1 ]]; then
        log "Skipping Linux Python environment provisioning as requested."
        return
    fi

    create_linux_venv

    log "Upgrading pip, wheel and setuptools"
    python -m pip install --upgrade pip wheel setuptools

    local requirements_file="${PROJECT_ROOT}/requirements.txt"
    if [[ -f "$requirements_file" ]]; then
        log "Installing project dependencies from requirements.txt (excluding MetaTrader5 wheel)"
        if grep -qi '^MetaTrader5' "$requirements_file"; then
            ensure_directory "$CACHE_DIR"
            grep -vi '^MetaTrader5' "$requirements_file" >"${CACHE_DIR}/requirements.nomt5.txt"
            python -m pip install --upgrade -r "${CACHE_DIR}/requirements.nomt5.txt"
        else
            python -m pip install --upgrade -r "$requirements_file"
        fi
    else
        warn "requirements.txt not found; skipping Python dependency installation."
    fi
}

write_env_file() {
    local win_py_path_escaped="${WIN_PY_WINDOWS_PATH:-}"
    if [[ -n "$win_py_path_escaped" ]]; then
        win_py_path_escaped="$(printf '%s' "$win_py_path_escaped" | sed 's/\\/\\\\/g')"
    fi

    local terminal_winpath
    terminal_winpath="$(to_windows_path "$MT5_TERMINAL")"

    cat >"$ENV_FILE" <<ENVFILE
# --- auto-generated by provision_mt5_environment.sh ---
WINEPREFIX=${MT5_WINE_PREFIX}
PYMT5LINUX_WINEPREFIX=${WIN_PY_WINE_PREFIX}
WIN_PY_WINE_PREFIX=${WIN_PY_WINE_PREFIX}
WINE_PYTHON=${win_py_path_escaped}
MT5_TERMINAL_PATH=${terminal_winpath}
ENVFILE
    log "Wrote Wine/MetaTrader environment configuration to $ENV_FILE"
}

write_login_instructions() {
    cat >"$LOGIN_INSTRUCTIONS" <<LOGINFILE
MetaTrader 5 (Wine) login & bridge instructions
=============================================

Terminal prefix : ${MT5_WINE_PREFIX}
Windows Python  : ${WIN_PY_WINDOWS_PATH:-<not detected>}
Terminal path   : ${MT5_TERMINAL}
Bridge exports  : export PYMT5LINUX_PYTHON="${WIN_PY_WINDOWS_PATH:-<not detected>}"
                  export PYMT5LINUX_WINEPREFIX="${WIN_PY_WINE_PREFIX}"

1. Launch the terminal and log in with your broker account:
   WINEARCH=win64 WINEPREFIX="${MT5_WINE_PREFIX}" wine "${MT5_TERMINAL}"
2. Re-run this script whenever you update the Wine prefix or upgrade Python.
3. Use scripts/setup_terminal.py for detailed diagnostics.
LOGINFILE
    log "Saved login instructions to $LOGIN_INSTRUCTIONS"
}

install_heartbeat_script() {
    if [[ "$NON_INTERACTIVE" -eq 1 ]]; then
        return
    fi
    local heartbeat_source="${PROJECT_ROOT}/mt5/mql5/ConnectionHeartbeat.mq5"
    if [[ ! -f "$heartbeat_source" ]]; then
        warn "Heartbeat script not found at ${heartbeat_source}."
        return
    fi

    local target_dir="${MT5_DIR}/MQL5/Scripts/MT5Bridge"
    ensure_directory "$target_dir"
    if cp "$heartbeat_source" "$target_dir/ConnectionHeartbeat.mq5"; then
        log "Installed ConnectionHeartbeat.mq5 into ${target_dir}"
    else
        warn "Failed to copy heartbeat script into ${target_dir}."
    fi
}

prompt_for_login() {
    if [[ "$NON_INTERACTIVE" -eq 1 ]]; then
        warn "Non-interactive mode enabled; skipping MetaTrader login prompt."
        return
    fi
    if [[ ! -t 0 ]]; then
        warn "Skipping MetaTrader login prompt because stdin is not a TTY."
        return
    fi
    if [[ ! -f "$MT5_TERMINAL" ]]; then
        warn "MetaTrader terminal binary not found at $MT5_TERMINAL; skipping login prompt."
        return
    fi

    log "Launching MetaTrader 5 terminal so you can log in. Close the terminal after it connects."
    WINEARCH=win64 WINEPREFIX="$MT5_WINE_PREFIX" wine "$MT5_TERMINAL" >/dev/null 2>&1 &
    local wine_pid=$!

    while true; do
        read -r -p "Type 'yes' after the terminal is logged in (or 'skip' to continue anyway): " answer || break
        case "${answer,,}" in
            yes)
                break
                ;;
            skip)
                warn "Continuing without confirming MetaTrader login."
                break
                ;;
            *)
                echo "Please respond with 'yes' after login or 'skip' to continue."
                ;;
        esac
    done

    if command_exists wineserver; then
        WINEPREFIX="$MT5_WINE_PREFIX" wineserver -k >/dev/null 2>&1 || true
    fi
    if kill -0 "$wine_pid" >/dev/null 2>&1; then
        wait "$wine_pid" || true
    fi
}

verify_mt5_bridge() {
    if [[ "$SKIP_CONNECTIVITY_CHECK" -eq 1 ]]; then
        log "Skipping MetaTrader 5 connectivity verification as requested."
        return
    fi

    if [[ ! -f "${PROJECT_ROOT}/scripts/setup_terminal.py" ]]; then
        warn "setup_terminal.py not found; cannot verify bridge."
        return
    fi

    log "Verifying MetaTrader 5 bridge from the Python environment"
    if command_exists python; then
        python "${PROJECT_ROOT}/scripts/setup_terminal.py" --install-heartbeat || warn "MetaTrader 5 connectivity check failed. Ensure the terminal is running and logged in."
    else
        warn "python command not available; skipping bridge verification."
    fi
}

install_services() {
    if [[ "$SKIP_SERVICE_INSTALL" -eq 1 ]]; then
        log "Skipping service installation as requested."
        return
    fi

    local service_file="${PROJECT_ROOT}/deploy/mt5bot.service"
    if [[ ! -f "$service_file" ]]; then
        warn "Service definitions not found; skipping systemd unit installation."
        return
    fi

    if ! command_exists systemctl; then
        warn "systemctl not available; skipping systemd unit installation."
        return
    fi

    log "Installing systemd services"
    run_with_sudo sed "s|{{REPO_PATH}}|${PROJECT_ROOT}|g" "${PROJECT_ROOT}/deploy/mt5bot.service" | run_with_sudo tee /etc/systemd/system/mt5bot.service >/dev/null
    if [[ -f "${PROJECT_ROOT}/deploy/mt5bot-update.service" ]]; then
        run_with_sudo sed "s|{{REPO_PATH}}|${PROJECT_ROOT}|g" "${PROJECT_ROOT}/deploy/mt5bot-update.service" | run_with_sudo tee /etc/systemd/system/mt5bot-update.service >/dev/null
    fi
    if [[ -f "${PROJECT_ROOT}/deploy/mt5bot-update.timer" ]]; then
        run_with_sudo sed "s|{{REPO_PATH}}|${PROJECT_ROOT}|g" "${PROJECT_ROOT}/deploy/mt5bot-update.timer" | run_with_sudo tee /etc/systemd/system/mt5bot-update.timer >/dev/null
    fi

    run_with_sudo systemctl daemon-reload
    run_with_sudo systemctl enable --now mt5bot.service || warn "Failed to enable mt5bot.service"
    if [[ -f "${PROJECT_ROOT}/deploy/mt5bot-update.timer" ]]; then
        run_with_sudo systemctl enable --now mt5bot-update.timer || warn "Failed to enable mt5bot-update.timer"
    fi
}

main() {
    log "Provisioning MetaTrader 5 environment from ${PROJECT_ROOT}"
    ensure_directory "$CACHE_DIR"

    ensure_apt_packages
    install_windows_python || warn "Windows Python provisioning reported errors"
    install_mt5_terminal || warn "MetaTrader terminal provisioning reported errors"
    install_mt5_windows_packages
    write_env_file
    write_login_instructions
    install_heartbeat_script
    prompt_for_login

    install_linux_requirements

    install_services
    verify_mt5_bridge

    log "Provisioning complete. Review warnings above for any manual follow-up."
}

main "$@"
