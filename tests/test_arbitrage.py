import sys
from pathlib import Path
import importlib.util

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)

spec = importlib.util.spec_from_file_location(
    "data.tick_aggregator", Path(__file__).resolve().parents[1] / "data" / "tick_aggregator.py"
)
ta = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ta
spec.loader.exec_module(ta)
compute_spread_matrix = ta.compute_spread_matrix

from strategy.arbitrage import detect_arbitrage
from strategy.router import StrategyRouter


def test_compute_spread_matrix():
    quotes = {
        "a": (1.0, 1.2),
        "b": (1.1, 1.3),
        "c": (1.05, 1.25),
    }
    matrix = compute_spread_matrix(quotes)
    assert matrix.shape == (3, 3)
    mid_a = 1.1
    mid_b = 1.2
    assert abs(matrix.loc["a", "b"] - abs(mid_a - mid_b)) < 1e-9
    assert matrix.loc["a", "a"] == 0


def test_detect_arbitrage(tmp_path, monkeypatch):
    router = StrategyRouter(algorithms={"arbitrage": lambda f: f.get("arbitrage", 0.0)})
    calls = []
    monkeypatch.setattr(router, "select", lambda f: calls.append(f))
    from strategy import arbitrage as arb_mod

    monkeypatch.setattr(arb_mod, "LOG_DIR", tmp_path)
    quotes = {"a": (1.0, 1.1), "b": (1.2, 1.3), "c": (1.05, 1.15)}
    sigs = detect_arbitrage("EURUSD", quotes, 0.1, router)
    # spreads: a-b=0.2, a-c=0.05, b-c=0.15 -> expect 2 signals
    assert len(sigs) == 2
    assert len(calls) == 2
    log_file = tmp_path / "signals.csv"
    assert log_file.exists()
    df = pd.read_csv(log_file)
    assert len(df) == 2
