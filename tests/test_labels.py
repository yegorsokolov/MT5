import pandas as pd
from pathlib import Path
import importlib.util
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

spec = importlib.util.spec_from_file_location(
    "labels", Path(__file__).resolve().parents[1] / "data" / "labels.py"
)
labels_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels_mod)
triple_barrier = labels_mod.triple_barrier


def test_triple_barrier_pt_hit():
    prices = pd.Series([100, 102, 101])
    labels = triple_barrier(prices, pt_mult=0.01, sl_mult=0.01, max_horizon=2)
    assert labels.iloc[0] == 1


def test_triple_barrier_sl_hit():
    prices = pd.Series([100, 98, 99])
    labels = triple_barrier(prices, pt_mult=0.01, sl_mult=0.01, max_horizon=2)
    assert labels.iloc[0] == -1


def test_triple_barrier_no_hit():
    prices = pd.Series([100, 100.5, 100.8])
    labels = triple_barrier(prices, pt_mult=0.05, sl_mult=0.05, max_horizon=2)
    assert labels.iloc[0] == 0
