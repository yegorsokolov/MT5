# Shared helpers for preparing the auxiliary mt5linux virtual environment.
#
# The functions defined here are intended to be sourced from provisioning
# scripts. They avoid mutating global shell options so they can be consumed from
# callers that already run with "set -euo pipefail".

# Resolve the repository root if the caller did not set PROJECT_ROOT.
if [[ -z ${PROJECT_ROOT:-} ]]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

: "${MT5LINUX_VENV_PATH:=${PROJECT_ROOT}/.mt5linux-venv}"
: "${MT5LINUX_LOCK_FILE:=${PROJECT_ROOT}/mt5linux-lock.txt}"

_mt5linux_env_log() {
    printf '[mt5linux-env] %s\n' "$*" >&2
}

_mt5linux_env_error() {
    printf '[mt5linux-env:ERROR] %s\n' "$*" >&2
}

_mt5linux_find_bootstrap_python() {
    local requested="$1"
    if [[ -n "$requested" ]]; then
        if command -v "$requested" >/dev/null 2>&1; then
            command -v "$requested"
            return 0
        fi
        if [[ -x "$requested" ]]; then
            printf '%s\n' "$requested"
            return 0
        fi
        _mt5linux_env_error "Bootstrap interpreter '$requested' not found"
        return 1
    fi

    local candidate
    for candidate in python3 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done

    _mt5linux_env_error "Unable to locate a Python 3 interpreter for the mt5linux auxiliary environment"
    return 1
}

refresh_mt5linux_venv() {
    local bootstrap="$1"
    local python_bin
    python_bin="$(_mt5linux_find_bootstrap_python "$bootstrap")" || return 1

    if [[ ! -d "$MT5LINUX_VENV_PATH" ]]; then
        _mt5linux_env_log "Creating mt5linux auxiliary virtualenv at $MT5LINUX_VENV_PATH"
        "$python_bin" -m venv "$MT5LINUX_VENV_PATH" || return 1
    fi

    # shellcheck disable=SC1090
    source "$MT5LINUX_VENV_PATH/bin/activate"
    python -m pip install --upgrade pip wheel setuptools || {
        deactivate >/dev/null 2>&1 || true
        return 1
    }

    if [[ -f "$MT5LINUX_LOCK_FILE" ]]; then
        _mt5linux_env_log "Replaying locked mt5linux dependencies from $(basename "$MT5LINUX_LOCK_FILE")"
        python -m pip install --upgrade -r "$MT5LINUX_LOCK_FILE"
    else
        _mt5linux_env_log "Lock file $MT5LINUX_LOCK_FILE missing; installing mt5linux and rpyc from PyPI"
        python -m pip install --upgrade mt5linux rpyc
    fi

    deactivate >/dev/null 2>&1 || true
    return 0
}

mt5linux_env_python_path() {
    if [[ -x "$MT5LINUX_VENV_PATH/bin/python" ]]; then
        printf '%s\n' "$MT5LINUX_VENV_PATH/bin/python"
        return 0
    fi
    return 1
}
