#!/usr/bin/env bash
set -euo pipefail

# -------- configurable inputs --------
PY_WIN_VERSION="3.11.9"
PY_WIN_DIR="C:\\Python311"                          # TargetDir for the EXE
PY_WIN_EXE_CACHE="/opt/mt5/.cache/mt5/python-${PY_WIN_VERSION}-amd64.exe"
PY_WIN_ZIP_CACHE="/opt/mt5/.cache/mt5/python-${PY_WIN_VERSION}-embed-amd64.zip"  # optional fallback
WINEPREFIX="${WINEPREFIX:-$HOME/.wine-py311}"
WINEARCH="${WINEARCH:-win64}"
USE_XVFB="${USE_XVFB:-1}"                           # set 0 if you have a GUI display
# -------------------------------------

log()  { printf "[setup] %s\n" "$*" >&2; }
fail() { printf "[setup:ERROR] %s\n" "$*" >&2; exit 1; }

# Prevent host venv from polluting anything
if [[ -n "${VIRTUAL_ENV-}" ]]; then
  log "Deactivating host Python venv ($VIRTUAL_ENV)"
  # shellcheck disable=SC1090
  deactivate 2>/dev/null || true
  unset VIRTUAL_ENV
fi
export PYTHONDONTWRITEBYTECODE=1

# Helper to run wine with or without xvfb
wrun() {
  if [[ "${USE_XVFB}" == "1" || -z "${DISPLAY-}" ]]; then
    xvfb-run -a --server-args="-screen 0 1280x800x24" wine "$@"
  else
    wine "$@"
  fi
}

# 1) Prepare Wine prefix
log "Initialising Wine prefix at $WINEPREFIX ..."
WINEDEBUG=-all WINEARCH="$WINEARCH" WINEPREFIX="$WINEPREFIX" wineboot -u
# Install VC++ runtime needed by Python 3.11
log "Ensuring VC++ runtime (vcrun2022) is installed ..."
# winetricks bails out if another VC runtime (e.g. vcrun2019) is present.
# Detect that situation and retry with --force so we don't fail setup when an
# older runtime already exists in the prefix.
if ! WINEPREFIX="$WINEPREFIX" winetricks -q vcrun2022 corefonts; then
  if WINEPREFIX="$WINEPREFIX" winetricks list-installed 2>/dev/null | grep -qi 'vcrun2022'; then
    log "VC++ runtime already present in prefix; continuing"
  elif WINEPREFIX="$WINEPREFIX" winetricks list-installed 2>/dev/null | grep -qi 'vcrun2019'; then
    log "vcrun2019 detected; retrying vcrun2022 install with --force"
    WINEPREFIX="$WINEPREFIX" winetricks --force -q vcrun2022 corefonts || \
      fail "Failed to install VC++ runtime (vcrun2022) even with --force"
  else
    fail "Failed to install VC++ runtime (vcrun2022)"
  fi
fi

# 2) Guard: cached installer must exist
[[ -f "$PY_WIN_EXE_CACHE" ]] || fail "Windows Python installer not found at $PY_WIN_EXE_CACHE"
# Optional: if you also cached the embeddable ZIP, we’ll use it only as fallback
[[ -f "$PY_WIN_ZIP_CACHE" ]] || true

# 3) If Python already present and runnable, we’re done
if WINEPREFIX="$WINEPREFIX" wrun "${PY_WIN_DIR}\\python.exe" -V >/dev/null 2>&1; then
  log "Windows Python already installed at ${PY_WIN_DIR}"
  exit 0
fi

# 4) Run the official EXE installer (quiet + pinned TargetDir)
log "Installing Windows Python ${PY_WIN_VERSION} ..."
INSTALL_ARGS=( "/quiet" "InstallAllUsers=1" "PrependPath=1" "TargetDir=${PY_WIN_DIR}" "Include_launcher=0" "Include_test=0" )
WINEPREFIX="$WINEPREFIX" wrun "$PY_WIN_EXE_CACHE" "${INSTALL_ARGS[@]}"

# 5) Verify install (primary)
if ! WINEPREFIX="$WINEPREFIX" wrun "${PY_WIN_DIR}\\python.exe" -V >/dev/null 2>&1; then
  log "EXE install did not produce ${PY_WIN_DIR}\\python.exe; trying embeddable ZIP fallback ..."
  # 5a) Fallback to embeddable ZIP if available
  if [[ -f "$PY_WIN_ZIP_CACHE" ]]; then
    # Expand the ZIP under C:\Python311
    # Use Windows cmd to create the target dir
    WINEPREFIX="$WINEPREFIX" wrun cmd /c "if not exist ${PY_WIN_DIR} mkdir ${PY_WIN_DIR}" || true
    # Unzip via PowerShell (present in recent Wine prefixes) or 7z if mapped; try both
    if ! WINEPREFIX="$WINEPREFIX" wrun powershell -NoProfile -Command \
        "Expand-Archive -Path 'Z:${PY_WIN_ZIP_CACHE//\//\\}' -DestinationPath '${PY_WIN_DIR}' -Force" >/dev/null 2>&1; then
      # Fallback to 7-Zip if you’ve installed it in Wine; otherwise bail
      fail "Failed to expand embeddable ZIP via PowerShell; install 7-Zip in Wine or pre-unpack."
    fi

    # Bootstrap pip into the embeddable build
    # The embeddable Python ships without pip; fetch get-pip.py into the prefix and run
    # If you don’t have network egress from Wine, copy a vendored get-pip.py into the cache and use that path.
    GETPIP_WIN="C:\\get-pip.py"
    if ! WINEPREFIX="$WINEPREFIX" wrun "${PY_WIN_DIR}\\python.exe" -m pip -V >/dev/null 2>&1; then
      log "Bootstrapping pip into embeddable Python ..."
      # Try to download get-pip.py using Linux curl then run it via Wine
      curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
      WINEPREFIX="$WINEPREFIX" wrun cmd /c "copy /Y Z:\\tmp\\get-pip.py ${GETPIP_WIN}" >/dev/null
      WINEPREFIX="$WINEPREFIX" wrun "${PY_WIN_DIR}\\python.exe" "${GETPIP_WIN}"
    fi
  else
    fail "No embeddable ZIP cached at $PY_WIN_ZIP_CACHE and EXE install failed."
  fi
fi

# 6) Final verification (must pass)
if ! WINEPREFIX="$WINEPREFIX" wrun "${PY_WIN_DIR}\\python.exe" -V; then
  fail "Windows Python install failed (EXE and fallback)."
fi

# 7) Make sure we can call pip and install a tiny wheel
WINEPREFIX="$WINEPREFIX" wrun "${PY_WIN_DIR}\\python.exe" -m pip install -U pip wheel setuptools
WINEPREFIX="$WINEPREFIX" wrun "${PY_WIN_DIR}\\python.exe" -m pip install "colorama==0.4.6"

log "Windows Python ${PY_WIN_VERSION} is installed and functional at ${PY_WIN_DIR}"
