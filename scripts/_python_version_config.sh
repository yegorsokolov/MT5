# shellcheck shell=bash
# Shared defaults for the Python interpreter versions used across onboarding scripts.
#
# MT5_PYTHON_SERIES controls the major.minor series (e.g. 3.13 or 3.11).
# MT5_PYTHON_PATCH may be set to pin an exact patch release. When unset we
# automatically choose a stable patch for the configured series.
#
# Scripts consuming these defaults may rely on the following variables:
#   MT5_PYTHON_SERIES        -> "3.13"
#   MT5_PYTHON_PATCH         -> "3.13.1" (or series-specific default)
#   MT5_PYTHON_TAG           -> "313" (digits only)
#   MT5_PYTHON_PREFIX        -> default Wine prefix ("$HOME/.wine-py313")
#   MT5_PYTHON_WIN_DIR       -> Windows installation target ("C:\\Python313")
#   MT5_PYTHON_INSTALLER     -> Windows installer filename
#   MT5_PYTHON_EMBED_ZIP     -> Optional embeddable ZIP filename
#   MT5_PYTHON_DOWNLOAD_ROOT -> Base URL for python.org artifacts

: "${MT5_PYTHON_SERIES:=3.13}"

if [[ -z "${MT5_PYTHON_PATCH:-}" ]]; then
  case "$MT5_PYTHON_SERIES" in
    3.11) MT5_PYTHON_PATCH="3.11.9" ;;
    3.13) MT5_PYTHON_PATCH="3.13.1" ;;
    *) MT5_PYTHON_PATCH="${MT5_PYTHON_SERIES}.0" ;;
  esac
fi

MT5_PYTHON_TAG="${MT5_PYTHON_SERIES//./}"
: "${MT5_PYTHON_PREFIX_NAME:=.wine-py${MT5_PYTHON_TAG}}"
: "${MT5_PYTHON_PREFIX:="$HOME/${MT5_PYTHON_PREFIX_NAME}"}"
: "${MT5_PYTHON_WIN_DIR:=C:\\Python${MT5_PYTHON_TAG}}"
MT5_PYTHON_INSTALLER="python-${MT5_PYTHON_PATCH}-amd64.exe"
MT5_PYTHON_EMBED_ZIP="python-${MT5_PYTHON_PATCH}-embed-amd64.zip"
MT5_PYTHON_DOWNLOAD_ROOT="https://www.python.org/ftp/python/${MT5_PYTHON_PATCH}"
