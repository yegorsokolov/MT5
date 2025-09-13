from importlib.metadata import EntryPoint
from pathlib import Path
import sys
import importlib.metadata


def test_plugins_register(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    plugin_root = root / "plugins" / "example_plugin"
    sys.path.extend([str(root), str(plugin_root)])

    def fake_entry_points(group=None):
        if group == "mt5.indicators":
            return [EntryPoint(
                name="plugin_indicator",
                value="example_plugin.indicator:register",
                group="mt5.indicators",
            )]
        if group == "mt5.strategies":
            return [EntryPoint(
                name="plugin_strategy",
                value="example_plugin.strategy:register",
                group="mt5.strategies",
            )]
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    import types
    data_pkg = types.ModuleType("data")
    data_features = types.ModuleType("data.features")
    data_features.make_features = lambda *a, **k: None
    data_pkg.features = data_features
    sys.modules.setdefault("data", data_pkg)
    sys.modules.setdefault("data.features", data_features)

    monkeypatch.setenv("MT5_DOCS_BUILD", "1")

    import features
    from src.strategy import registry as strat_registry

    pipeline = features.get_feature_pipeline()
    assert any(fn.__module__ == "example_plugin.indicator" for fn in pipeline)

    strat = strat_registry.get_strategy("plugin_strategy")
    assert strat["name"] == "plugin_strategy"

