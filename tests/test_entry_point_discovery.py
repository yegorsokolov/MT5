"""Tests for entry-point driven plugin discovery."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))


@dataclass
class DummyEntryPoint:
    """Minimal stub mimicking ``importlib.metadata.EntryPoint``."""

    name: str
    loader: Callable[[], Callable[[Callable[..., Any]], None]]

    def load(self) -> Callable[[Callable[..., Any]], None]:
        return self.loader()


def _entry_point_factory(entries: Dict[str, List[DummyEntryPoint]]):
    """Return an ``entry_points`` replacement for the given groups."""

    def _entry_points(*args: Any, **kwargs: Any) -> Iterable[DummyEntryPoint] | Dict[str, List[DummyEntryPoint]]:
        if kwargs.get("group"):
            return entries.get(kwargs["group"], [])
        if not args and not kwargs:
            return entries
        return []

    return _entry_points


def test_feature_entry_points_register(monkeypatch):
    monkeypatch.setenv("MT5_DOCS_BUILD", "1")
    data_pkg = types.ModuleType("data")
    data_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "data", data_pkg)
    data_features_pkg = types.ModuleType("data.features")
    data_features_pkg.make_features = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "data.features", data_features_pkg)

    import features

    plugin_name = "entrypoint_feature"

    def hook(register: Callable[..., None]) -> None:
        def compute(df):  # pragma: no cover - simple passthrough
            return df

        register(plugin_name, compute)

    entries = {"mt5.features": [DummyEntryPoint("dummy_feature", lambda: hook)]}
    monkeypatch.setattr(features, "entry_points", _entry_point_factory(entries))

    features._REGISTRY.pop(plugin_name, None)
    features._external_loaded = False

    try:
        features._load_external_features()
        assert plugin_name in features._REGISTRY
    finally:
        features._REGISTRY.pop(plugin_name, None)
        features._external_loaded = False
        sys.modules.pop("features", None)
        sys.modules.pop("features.validators", None)


def test_strategy_entry_points_register(monkeypatch):
    import strategies

    plugin_name = "entrypoint_strategy"

    def hook(register: Callable[..., None]) -> None:
        register(plugin_name, lambda **_: object(), description="via entry point")

    entries = {"mt5.strategies": [DummyEntryPoint("dummy_strategy", lambda: hook)]}
    monkeypatch.setattr(strategies, "entry_points", _entry_point_factory(entries))

    strategies._REGISTRY.pop(plugin_name, None)
    strategies._external_loaded = False

    try:
        available = strategies.available_strategies()
        assert plugin_name in available
    finally:
        strategies._REGISTRY.pop(plugin_name, None)
        strategies._external_loaded = False
