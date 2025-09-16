import time

import time

import time

import pandas.testing as pdt
import numpy as np
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pandas as pd
import polars as pl


def _load_feature_mod(name: str):
    base = Path(__file__).resolve().parents[1] / "features" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, base)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


order_flow = _load_feature_mod("order_flow")
baseline_signal = _load_feature_mod("baseline_signal")


def _random_order_flow(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "bid_sz_0": rng.integers(1, 100, size=n),
            "ask_sz_0": rng.integers(1, 100, size=n),
        }
    )


def test_order_flow_parity():
    pdf = _random_order_flow(100)
    pd_res = order_flow.compute(pdf.copy())
    pl_res = order_flow.compute(pl.from_pandas(pdf))
    pdt.assert_frame_equal(pl_res.to_pandas(), pd_res)


def test_baseline_signal_parity():
    data = pd.DataFrame(
        {
            "Close": np.linspace(1, 10, 50),
            "High": np.linspace(1, 10, 50) + 0.5,
            "Low": np.linspace(1, 10, 50) - 0.5,
        }
    )
    pd_res = baseline_signal.compute(data.copy())
    pl_res = baseline_signal.compute(pl.from_pandas(data))
    cols = ["baseline_signal", "long_stop", "short_stop", "baseline_confidence"]
    pdt.assert_frame_equal(pl_res.to_pandas()[cols], pd_res[cols])


def test_order_flow_speed():
    pdf = _random_order_flow(10000)
    start = time.time()
    order_flow.compute(pdf.copy())
    pandas_time = time.time() - start

    start = time.time()
    order_flow.compute(pl.from_pandas(pdf))
    polars_time = time.time() - start

    assert polars_time <= pandas_time * 1.5
