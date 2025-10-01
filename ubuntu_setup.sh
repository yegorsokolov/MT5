#!/usr/bin/env bash
set -Eeuo pipefail

log_dir="/opt/mt5"
log_file="${log_dir}/setup.log"
cache_dir="${log_dir}/.cache/mt5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ASSETS_DIR="${SCRIPT_DIR}/mt5_bridge_files"
mt5_cache_installer="${cache_dir}/mt5linux.sh"
python_installer="${cache_dir}/python-3.11.9-amd64.exe"
required_packages=("winehq-stable" "winetricks" "xvfb" "cabextract" "p7zip-full" "curl" "wget" "rsync" "jq" "unzip")

FORCE_INSTALLER_REFRESH="${FORCE_INSTALLER_REFRESH:-0}"

PY_PREFIX="${HOME}/.wine-py311"
FETCH_PREFIX="${HOME}/.mt5"
MT5_INSTALL_SUBPATH="drive_c/Program Files/MetaTrader 5"
PYTHON_WIN_PATH="C:\\Python311\\python.exe"
MT5_TERMINAL="C:\\Program Files\\MetaTrader 5\\terminal64.exe"
MT5_METAEDITOR="C:\\Program Files\\MetaTrader 5\\metaeditor64.exe"
BRIDGE_SUBDIR="MQL5/Files/bridge"
BRIDGE_EA_DIR="MQL5/Experts/Bridge"
BRIDGE_EA_NAME="BridgeEA"
BRIDGE_EA_SOURCE="${BRIDGE_EA_DIR}/${BRIDGE_EA_NAME}.mq5"
BRIDGE_EA_COMPILED="${BRIDGE_EA_DIR}/${BRIDGE_EA_NAME}.ex5"
CONFIG_SCRIPT_WIN="C:\\Program Files\\MetaTrader 5\\MQL5\\Scripts\\Bridge\\AutoAttach.mq5"
CONFIG_SCRIPT_COMPILED_WIN="C:\\Program Files\\MetaTrader 5\\MQL5\\Scripts\\Bridge\\AutoAttach.ex5"
BRIDGE_CLIENT="${log_dir}/bridge_client.py"
BRIDGE_BASE_WIN="C:\\Program Files\\MetaTrader 5\\${BRIDGE_SUBDIR//\//\\}"
BRIDGE_BASE_LINUX="${PY_PREFIX}/${MT5_INSTALL_SUBPATH}/${BRIDGE_SUBDIR}"

WINE="wine"
WINESERVER="wineserver"
WINETRICKS="winetricks"
XVFB_RUN="xvfb-run -a"

ensure_dirs() {
  if [[ $(id -u) -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "This script requires root privileges to create ${log_dir}. Please run as root or install sudo." >&2
      exit 1
    fi
    sudo mkdir -p "${log_dir}"
    sudo chown "${USER}:${USER}" "${log_dir}"
  else
    mkdir -p "${log_dir}"
    chown "${USER}:${USER}" "${log_dir}"
  fi
  mkdir -p "${cache_dir}"
}

setup_logging() {
  touch "${log_file}"
  exec > >(tee -a "${log_file}") 2>&1
  echo "==== MetaTrader 5 Setup $(date -Iseconds) ===="
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

install_packages() {
  echo "Installing required packages via apt..."
  require_command apt-get
  if [[ $(id -u) -ne 0 ]]; then
    sudo apt-get update
    sudo apt-get install -y "${required_packages[@]}"
  else
    apt-get update
    apt-get install -y "${required_packages[@]}"
  fi
}

prepare_prefix() {
  local prefix="$1"
  local arch="win64"
  if [[ ! -d "${prefix}" ]]; then
    echo "Creating Wine prefix ${prefix}..."
    WINEPREFIX="${prefix}" WINEARCH="${arch}" ${WINE} wineboot -u
  else
    echo "Updating Wine prefix ${prefix}..."
    WINEPREFIX="${prefix}" WINEARCH="${arch}" ${WINE} wineboot -u
  fi
  echo "Setting Windows version to win10 for ${prefix}"
  WINEPREFIX="${prefix}" ${WINE} reg add 'HKCU\\Software\\Wine\\WineLt' /f >/dev/null 2>&1 || true
  WINEPREFIX="${prefix}" ${WINE} reg add 'HKCU\\Software\\Wine\\Drivers' /v Audio /t REG_SZ /d pulse /f >/dev/null 2>&1 || true
  WINEPREFIX="${prefix}" ${WINETRICKS} -q win10 || true
}

download_file() {
  local url="$1"
  local target="$2"
  local tmp="${target}.tmp"
  if [[ -s "${target}" ]]; then
    if [[ "${FORCE_INSTALLER_REFRESH}" != "0" ]]; then
      echo "Refreshing ${target} as FORCE_INSTALLER_REFRESH is set."
      rm -f "${target}" "${tmp}" || true
    else
      echo "File ${target} already exists and is non-empty; skipping download."
      return
    fi
  fi
  echo "Downloading ${url} -> ${target}"
  curl -fsSL "${url}" -o "${tmp}"
  mv "${tmp}" "${target}"
}

run_mt5_installer() {
  echo "Running official MT5 installer script..."
  chmod +x "${mt5_cache_installer}"
  if [[ $(id -u) -ne 0 ]]; then
    sudo env WINEPREFIX="${FETCH_PREFIX}" bash "${mt5_cache_installer}" -q || true
  else
    env WINEPREFIX="${FETCH_PREFIX}" bash "${mt5_cache_installer}" -q || true
  fi
}

find_terminal_path() {
  local candidates=(
    "${FETCH_PREFIX}/${MT5_INSTALL_SUBPATH}/terminal64.exe"
    "${HOME}/.mt5/${MT5_INSTALL_SUBPATH}/terminal64.exe"
  )
  for p in "${candidates[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

sync_terminal_to_python_prefix() {
  local source_dir="$1"
  local target_dir="${PY_PREFIX}/${MT5_INSTALL_SUBPATH}"
  mkdir -p "${target_dir}"
  echo "Syncing MT5 from ${source_dir} to ${target_dir}"
  rsync -a --delete "${source_dir}/" "${target_dir}/"
}

install_windows_python() {
  echo "Installing Windows Python if necessary..."
  local python_check
  if python_check=$(WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" -V 2>/dev/null); then
    echo "Windows Python already installed: ${python_check}"
    return
  fi
  echo "Running Python installer under Wine..."
  WINEPREFIX="${PY_PREFIX}" ${WINE} "${python_installer}" /quiet InstallAllUsers=1 PrependPath=1 TargetDir=C:\\Python311 Include_launcher=0 Include_test=0
  WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" -V
}

install_pip_packages() {
  local bridge_backend="${MT5_BRIDGE_BACKEND:-}"
  bridge_backend="${bridge_backend,,}"
  if [[ "${bridge_backend}" == "mql5" || "${bridge_backend}" == "grpc" ]]; then
    echo "Skipping Windows MetaTrader5 pip packages (bridge backend: ${bridge_backend})."
    return
  fi

  echo "Ensuring required Windows pip packages..."
  WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" -m pip install --upgrade pip

  local mt5_packages=("MetaTrader5")
  if ! WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" -m pip install --only-binary :all: "${mt5_packages[@]}"; then
    echo "Falling back to default installation for ${mt5_packages[*]}" >&2
    if ! WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" -m pip install "${mt5_packages[@]}"; then
      echo "Failed to install Windows MetaTrader5 dependencies" >&2
      exit 1
    fi
  fi
}

apply_winetricks() {
  echo "Applying winetricks components..."
  WINEPREFIX="${PY_PREFIX}" ${WINETRICKS} -q vcrun2022 winhttp wininet corefonts
  echo "Adding DLL overrides for winhttp/wininet"
  WINEPREFIX="${PY_PREFIX}" ${WINE} reg add 'HKCU\\Software\\Wine\\DllOverrides' /v winhttp /t REG_SZ /d n,b /f
  WINEPREFIX="${PY_PREFIX}" ${WINE} reg add 'HKCU\\Software\\Wine\\DllOverrides' /v wininet /t REG_SZ /d n,b /f
  for dll in ucrtbase msvcp140 vcruntime140; do
    WINEPREFIX="${PY_PREFIX}" ${WINE} reg add 'HKCU\\Software\\Wine\\DllOverrides' /v "${dll}" /t REG_SZ /d native,builtin /f || true
  done
}

start_mt5_headless_once() {
  echo "Performing initial MT5 headless launch..."
  local exe="${MT5_TERMINAL}"
  WINEPREFIX="${PY_PREFIX}" ${XVFB_RUN} ${WINE} "${exe}" /portable /log /skipupdate || true &
  local pid=$!
  sleep 25
  if ps -p ${pid} >/dev/null 2>&1; then
    echo "Initial MT5 launch still running; leaving for background initialization."
  else
    echo "Initial MT5 launch finished."
  fi
}

compile_ea() {
  echo "Compiling BridgeEA via MetaEditor..."
  WINEPREFIX="${PY_PREFIX}" ${XVFB_RUN} ${WINE} "${MT5_METAEDITOR}" /log /compile:"C:\\Program Files\\MetaTrader 5\\${BRIDGE_EA_SOURCE//\//\\}"
}

compile_auto_attach() {
  echo "Compiling AutoAttach script..."
  WINEPREFIX="${PY_PREFIX}" ${XVFB_RUN} ${WINE} "${MT5_METAEDITOR}" /log /compile:"${CONFIG_SCRIPT_WIN}"
}

deploy_bridge_files() {
  if [[ ! -d "${BRIDGE_ASSETS_DIR}" ]]; then
    echo "Bridge asset directory ${BRIDGE_ASSETS_DIR} missing" >&2
    exit 1
  fi
  local target_root="${PY_PREFIX}/${MT5_INSTALL_SUBPATH}/MQL5"
  mkdir -p "${target_root}"
  echo "Copying bridge assets from ${BRIDGE_ASSETS_DIR} to ${target_root}"
  rsync -a "${BRIDGE_ASSETS_DIR}/" "${target_root}/"
}

write_bridge_client() {
  cat >"${BRIDGE_CLIENT}" <<PYEOF
#!/usr/bin/env python3
import json
import os
import sys
import time
import uuid
from pathlib import Path

BASE = Path(os.environ.get("MT5_BRIDGE_BASE", "${BRIDGE_BASE_LINUX}"))
COMMAND = BASE / "command.json"
RESPONSE = BASE / "response.json"
TICK = BASE / "tick.json"

TIMEOUT = 30


def ensure_base():
    BASE.mkdir(parents=True, exist_ok=True)
    return BASE


def send_ping():
    ensure_base()
    cmd_id = uuid.uuid4().hex
    COMMAND.write_text(f"{cmd_id}:ping", encoding="utf-8")
    start = time.time()
    while time.time() - start < TIMEOUT:
        if RESPONSE.exists():
            try:
                data = json.loads(RESPONSE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(0.5)
                continue
            if data.get("id") == cmd_id:
                return data
        time.sleep(0.5)
    raise TimeoutError("No response from bridge EA")


def read_tick():
    if not TICK.exists():
        return None
    try:
        return json.loads(TICK.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main():
    try:
        pong = send_ping()
        tick = read_tick()
    except Exception as exc:  # noqa: BLE001
        print("bridge_client: FAIL", exc)
        print("Hint: ensure MT5 terminal is running with BridgeEA attached.")
        sys.exit(1)
    print("bridge_client: PASS")
    print(json.dumps({"pong": pong, "tick": tick}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
PYEOF
  chmod +x "${BRIDGE_CLIENT}"
}

launch_terminal_for_bridge() {
  echo "Launching MT5 with bridge profile..."
  local exe="${MT5_TERMINAL}"
  WINEPREFIX="${PY_PREFIX}" ${XVFB_RUN} ${WINE} "${exe}" /portable /log /skipupdate /script:"${CONFIG_SCRIPT_COMPILED_WIN}" || true &
  sleep 25
}

run_windows_python_check() {
  echo "Running MetaTrader5 Python verification..."
  cat >"${log_dir}/mt5_verify.py" <<'PYCHK'
import MetaTrader5 as mt5
print("INIT", mt5.initialize(timeout=180000))
print("ERR", mt5.last_error())
mt5.shutdown()
PYCHK
  WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" "Z:${log_dir}/mt5_verify.py"
}

run_bridge_client_test() {
  echo "Running Linux bridge client test..."
  python3 "${BRIDGE_CLIENT}" || true
}

main() {
  for arg in "$@"; do
    case "${arg}" in
      --fresh-installers)
        FORCE_INSTALLER_REFRESH=1
        ;;
    esac
  done

  ensure_dirs
  setup_logging
  for pkg in wget curl tar gzip; do
    require_command "${pkg}"
  done
  install_packages
  prepare_prefix "${PY_PREFIX}"
  prepare_prefix "${FETCH_PREFIX}"
  download_file "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5linux.sh" "${mt5_cache_installer}"
  download_file "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" "${python_installer}"
  run_mt5_installer
  local terminal_source
  if ! terminal_source=$(find_terminal_path); then
    echo "Failed to locate terminal64.exe after installation" >&2
    exit 1
  fi
  sync_terminal_to_python_prefix "$(dirname "${terminal_source}")"
  install_windows_python
  install_pip_packages
  apply_winetricks
  deploy_bridge_files
  compile_ea
  compile_auto_attach
  write_bridge_client
  start_mt5_headless_once
  launch_terminal_for_bridge
  run_windows_python_check || true
  run_bridge_client_test

  echo "Collecting final status..."
  local term_path="${PY_PREFIX}/${MT5_INSTALL_SUBPATH}/terminal64.exe"
  local python_version
  if python_version=$(WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" -V 2>/dev/null); then
    python_status="PASS (${python_version})"
  else
    python_status="FAIL"
  fi
  local mt5_pkg
  if mt5_pkg=$(WINEPREFIX="${PY_PREFIX}" ${WINE} "${PYTHON_WIN_PATH}" -m pip show MetaTrader5 2>/dev/null); then
    pip_status="PASS"
  else
    pip_status="FAIL"
  fi
  local bridge_ex5_status="FAIL"
  if [[ -f "${PY_PREFIX}/${MT5_INSTALL_SUBPATH}/${BRIDGE_EA_COMPILED}" ]]; then
    bridge_ex5_status="PASS"
  fi
  local bridge_ping_status="FAIL"
  if python3 "${BRIDGE_CLIENT}" >/dev/null 2>&1; then
    bridge_ping_status="PASS"
  fi

  echo "==== SUMMARY ===="
  if [[ -f "${term_path}" ]]; then
    echo "PASS: terminal64.exe located at ${term_path}"
  else
    echo "FAIL: terminal64.exe missing at ${term_path}"
  fi
  echo "Python check: ${python_status}"
  echo "pip show MetaTrader5: ${pip_status}"
  echo "Bridge EA compiled: ${bridge_ex5_status}"
  echo "bridge_client.py ping: ${bridge_ping_status}"

  if [[ ${bridge_ping_status} != "PASS" || ! -f "${term_path}" ]]; then
    exit 1
  fi

  echo "Next steps:"
  echo " - Start terminal: WINEPREFIX=${PY_PREFIX} ${WINE} \"${MT5_TERMINAL}\" /portable /log /skipupdate"
  echo " - Stop Wine services: WINEPREFIX=${PY_PREFIX} ${WINESERVER} -k -w"
  echo "Logs available at ${log_file}"
}

main "$@"
