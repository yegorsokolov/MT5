"""Unified entry point for the Adaptive MT5 toolkit.

This module acts as a small dispatcher so that operators can simply execute

``python -m mt5 [mode] [<module arguments>]``

and have the appropriate historical backtest, offline training, or realtime
trainer bootstrapped automatically.  The dispatcher keeps the legacy module
entry points (``mt5.train``, ``mt5.backtest`` and ``mt5.realtime_train``) but
allows running everything from a single executable which was a frequent request
from operations.

The selection logic honours multiple layers so it can integrate smoothly with
automation:

1. Command line arguments (either the positional ``mode`` argument or
   ``--mode`` option).
2. Environment variables ``MT5_MODE`` or ``MT5_DEFAULT_MODE``.
3. Configuration keys such as ``run_mode`` or ``runtime.mode`` from
   :func:`utils.load_config`.
4. Finally a safe fallback to the classic training pipeline.

The dispatcher executes the legacy modules in-process using ``runpy`` so the
behaviour and argument handling of the underlying modules remains unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - pydantic is an optional dependency in tests
    BaseModel = object  # type: ignore

try:
    from utils import load_config as _load_config
except Exception:  # pragma: no cover - optional dependency missing in tests
    _load_config = None


@dataclass(frozen=True)
class EntryPoint:
    """Descriptor for each supported mode."""

    module: str
    description: str


ENTRY_POINTS: dict[str, EntryPoint] = {
    "train": EntryPoint(
        module="mt5.train",
        description="Classic tabular training pipeline",
    ),
    "backtest": EntryPoint(
        module="mt5.backtest",
        description="Historical backtesting suite",
    ),
    "realtime": EntryPoint(
        module="mt5.realtime_train",
        description="Realtime / live training service",
    ),
}


ALIASES: dict[str, str] = {
    "training": "train",
    "classic": "train",
    "offline": "train",
    "backtesting": "backtest",
    "bt": "backtest",
    "realtime_train": "realtime",
    "realtime-train": "realtime",
    "real_time": "realtime",
    "live": "realtime",
    "live_train": "realtime",
    "live-train": "realtime",
}


def _normalise_mode(value: str | None) -> str | None:
    if not value:
        return None
    normalised = value.strip().lower().replace(" ", "_")
    normalised = normalised.replace("-", "_")
    if not normalised:
        return None
    if normalised in ENTRY_POINTS:
        return normalised
    return ALIASES.get(normalised)


def _extract_from_config(cfg: object) -> str | None:
    """Attempt to read the configured run mode from ``cfg``."""

    def _from_mapping(mapping: Mapping[str, object]) -> str | None:
        for key in ("mode", "run_mode", "entry_point", "entrypoint"):
            candidate = mapping.get(key)
            if isinstance(candidate, str):
                return candidate
        for nested_key in ("runtime", "orchestration"):
            nested = mapping.get(nested_key)
            if isinstance(nested, Mapping):
                nested_candidate = _from_mapping(nested)
                if nested_candidate:
                    return nested_candidate
        return None

    if isinstance(cfg, Mapping):
        return _from_mapping(cfg)

    if isinstance(cfg, BaseModel):  # type: ignore[isinstance]
        raw = cfg.model_dump()  # type: ignore[attr-defined]
        if isinstance(raw, Mapping):
            return _from_mapping(raw)
        return None

    getter = getattr(cfg, "get", None)
    if callable(getter):
        for key in ("run_mode", "mode", "entry_point", "entrypoint"):
            candidate = getter(key)  # type: ignore[call-arg]
            if isinstance(candidate, str):
                return candidate
    return None


def _select_mode(
    cli_mode: str | None,
    env: Mapping[str, str],
    config_sources: Sequence[object],
) -> str:
    """Determine which entry point should be executed."""

    for value in (cli_mode, env.get("MT5_MODE"), env.get("MT5_DEFAULT_MODE")):
        mode = _normalise_mode(value)
        if mode:
            return mode

    for cfg in config_sources:
        mode = _normalise_mode(_extract_from_config(cfg))
        if mode:
            return mode

    return "train"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mt5",
        description="Unified MT5 command dispatcher",
        add_help=True,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=sorted(ENTRY_POINTS),
        help="Target mode to execute (defaults to automatic selection)",
    )
    parser.add_argument(
        "--mode",
        dest="mode_option",
        choices=sorted(ENTRY_POINTS),
        help="Explicitly select a mode overriding auto-detection",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available modes and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resolved entry point without executing it",
    )
    return parser


def _run_module(module: str, argv: Iterable[str]) -> None:
    previous_argv = sys.argv[:]
    sys.argv = [module, *argv]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous_argv


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv if argv is not None else sys.argv[1:])
    parser = _build_parser()
    parsed, remainder = parser.parse_known_args(args_list)

    if parsed.list:
        for name in sorted(ENTRY_POINTS):
            entry = ENTRY_POINTS[name]
            print(f"{name:10s} -> {entry.module} ({entry.description})")
        return 0

    cli_mode = parsed.mode_option or parsed.mode

    config_candidates: list[object] = []
    if _load_config is not None:
        try:
            config_candidates.append(_load_config())
        except Exception:
            # Silently ignore configuration errors here; the downstream command
            # will surface issues when it actually runs.
            pass

    resolved_mode = _select_mode(cli_mode, os.environ, tuple(config_candidates))
    entry = ENTRY_POINTS.get(resolved_mode)
    if entry is None:
        parser.error(
            f"Unknown mode '{resolved_mode}'. Use --list to inspect supported modes."
        )

    if importlib.util.find_spec(entry.module) is None:
        parser.error(
            (
                f"Entry point module '{entry.module}' could not be imported. "
                "Ensure the optional component is installed before running this mode."
            )
        )

    if parsed.dry_run:
        print(f"{resolved_mode}: {entry.module}")
        return 0

    _run_module(entry.module, remainder)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m mt5
    sys.exit(main())

