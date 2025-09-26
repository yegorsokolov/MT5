#!/usr/bin/env python3
"""Detect the MetaTrader 5 terminal installation directory.

The helper script searches common installation locations for the MetaTrader 5
terminal executable. When it finds a candidate it writes the corresponding path
into a ``.env``-style file so the rest of the toolchain can read
``MT5_TERMINAL_PATH`` automatically.

Run the script from the project root:

```
python scripts/detect_mt5_terminal.py
```

Use ``--env-file`` to update a custom environment file instead of the default
``.env`` next to this repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_KEY = "MT5_TERMINAL_PATH"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _resolve_terminal_path(candidate: Path) -> Optional[Path]:
    """Return the MetaTrader 5 terminal executable if it exists under *candidate*."""

    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        return None

    for name in ("terminal64.exe", "terminal.exe", "terminal"):
        guess = candidate / name
        if guess.exists():
            return guess

    for child in candidate.iterdir():
        if child.is_file() and child.name.lower().startswith("terminal"):
            return child
    return None


def _candidate_directories() -> Iterable[Path]:
    """Yield directories that may contain the MetaTrader 5 terminal."""

    env_value = os.getenv(ENV_KEY)
    if env_value:
        yield Path(env_value)

    yield PROJECT_ROOT / "mt5"
    yield PROJECT_ROOT / "MT5"
    yield Path("/opt/mt5")

    home = Path.home()
    possible = [
        home / "MetaTrader 5",
        home / "Applications/MetaTrader 5.app/Contents/MacOS",
        home / "Applications/MetaTrader 5.app/Contents/Resources/drive_c/Program Files/MetaTrader 5",
        home / ".wine/drive_c/Program Files/MetaTrader 5",
        home / ".wine/drive_c/Program Files/MetaTrader 5 Terminal",
        home / ".wine/drive_c/Program Files (x86)/MetaTrader 5",
        home / ".wine/drive_c/Program Files (x86)/MetaTrader 5 Terminal",
        home / "AppData/Roaming/MetaQuotes/Terminal",
        home / "AppData/Local/Programs/MetaTrader 5",
    ]

    for directory in possible:
        if directory.exists():
            yield directory


def _discover_terminal() -> Optional[Path]:
    """Return the directory containing the MetaTrader 5 terminal."""

    for directory in _candidate_directories():
        terminal = _resolve_terminal_path(directory)
        if terminal is None:
            continue
        if terminal.parent:
            return terminal.parent
    return None


def _format_env_line(value: str) -> str:
    return f"{ENV_KEY}={value}"


def _write_env_file(path: Path, value: str) -> None:
    """Persist the resolved environment variable into ``path``."""

    if not path.exists():
        path.write_text(_format_env_line(value) + "\n", encoding="utf-8")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, _ = line.partition("=")
        if key == ENV_KEY:
            lines[index] = _format_env_line(value)
            updated = True
            break
    if not updated:
        lines.append(_format_env_line(value))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Environment file to update (default: %(default)s)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the detected path without modifying any files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    terminal_dir = _discover_terminal()
    if terminal_dir is None:
        print("Could not locate a MetaTrader 5 terminal. Set MT5_TERMINAL_PATH manually.")
        return 1

    resolved_value = str(terminal_dir)
    if args.print_only:
        print(resolved_value)
        return 0

    env_file = args.env_file
    try:
        _write_env_file(env_file, resolved_value)
    except OSError as exc:
        print(f"Failed to update {env_file}: {exc}")
        return 1

    print(f"Detected MetaTrader 5 terminal at {resolved_value}")
    print(f"Updated {env_file} with {ENV_KEY}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
