import pandas as pd
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "labels", Path(__file__).resolve().parents[1] / "data" / "labels.py"
)
labels_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels_mod)
labels_mod.log_lineage = lambda *a, **k: None
triple_barrier = labels_mod.triple_barrier


class SimpleCfg(dict):
    def get(self, key, default=None):
        training = dict.get(self, "training", {})
        strategy = dict.get(self, "strategy", {})
        if key in training:
            return training[key]
        if key in strategy:
            return strategy[key]
        return dict.get(self, key, default)


def test_triple_barrier_config_parity():
    cfg = SimpleCfg(
        {
            "training": {"pt_mult": 0.02, "sl_mult": 0.03, "max_horizon": 4},
            "strategy": {"symbols": ["T"], "risk_per_trade": 0.01},
        }
    )
    prices = pd.Series([100, 101, 102, 103, 104])
    expected = triple_barrier(prices, 0.02, 0.03, 4)
    result = triple_barrier(
        prices,
        cfg.get("pt_mult"),
        cfg.get("sl_mult"),
        cfg.get("max_horizon"),
    )
    assert result.equals(expected)
