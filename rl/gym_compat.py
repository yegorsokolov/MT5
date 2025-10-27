"""Compatibility helpers for the Gymnasium migration."""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from types import ModuleType, SimpleNamespace


class _GymStub(SimpleNamespace):
    """Fallback object that mimics the legacy truthiness behaviour."""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return False

__all__ = ["gym", "spaces"]


def _load_module(name: str) -> ModuleType | None:
    """Return the imported module when it can be resolved."""

    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    module = importlib.import_module(name)
    return module


def _resolve_gym_modules() -> tuple[ModuleType | SimpleNamespace, ModuleType | SimpleNamespace]:
    """Prefer Gymnasium while remaining compatible with legacy Gym."""

    gymnasium_module = _load_module("gymnasium")
    if gymnasium_module is not None:
        spaces_module = getattr(gymnasium_module, "spaces", SimpleNamespace())
        return gymnasium_module, spaces_module

    legacy_gym = _load_module("gym")
    if legacy_gym is not None:
        warnings.warn(
            "Gymnasium is not installed; falling back to the unmaintained 'gym' package.",
            RuntimeWarning,
            stacklevel=2,
        )
        spaces_module = getattr(legacy_gym, "spaces", SimpleNamespace())
        return legacy_gym, spaces_module

    warnings.warn(
        "Gymnasium is not installed and legacy 'gym' is unavailable; using a minimal stub.",
        RuntimeWarning,
        stacklevel=2,
    )
    stub = _GymStub(Env=object)
    return stub, SimpleNamespace()


gym, spaces = _resolve_gym_modules()

