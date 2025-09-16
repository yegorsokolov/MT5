import importlib.util
from pathlib import Path
import sys
import time
import types

import numpy as np
import pandas as pd

utils_stub = types.ModuleType("utils")


def _dummy_load_config():
    class _Cfg:
        strategy = types.SimpleNamespace(
            session_position_limits={}, default_position_limit=1
        )

    return _Cfg()


utils_stub.load_config = _dummy_load_config
sys.modules.setdefault("utils", utils_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _field(default=None, **kwargs):
    factory = kwargs.get("default_factory")
    if factory is not None:
        try:
            return factory()
        except TypeError:
            return factory
    if default is Ellipsis:
        return None
    return default


class _ConfigDict(dict):
    pass


pydantic_stub.BaseModel = _BaseModel
pydantic_stub.Field = _field
pydantic_stub.ConfigDict = _ConfigDict
sys.modules.setdefault("pydantic", pydantic_stub)
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
spec = importlib.util.spec_from_file_location(
    "baseline_signal", ROOT / "features" / "baseline_signal.py"
)
baseline_signal = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline_signal)

from strategies.baseline import BaselineStrategy, IndicatorBundle


def _sequential_reference(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    working = df.copy()
    strat = BaselineStrategy(
        **params, session_position_limits={}, default_position_limit=1
    )

    signals: list[float] = []
    long_stops: list[float] = []
    short_stops: list[float] = []
    confidences: list[float] = []

    for row in working.itertuples():
        price_val = row.Close
        indicator_row = IndicatorBundle(
            high=row.High if not pd.isna(row.High) else price_val,
            low=row.Low if not pd.isna(row.Low) else price_val,
        )
        sig = strat.update(price_val, indicator_row)
        signals.append(sig)

        if (
            strat.position == 1
            and strat.entry_price is not None
            and strat.entry_atr is not None
        ):
            peak = strat.peak_price if strat.peak_price is not None else price_val
            long_stop = max(
                strat.entry_price - strat.entry_atr * strat.atr_stop_long,
                peak * (1 - strat.trailing_stop_pct),
            )
        else:
            long_stop = np.nan

        if (
            strat.position == -1
            and strat.entry_price is not None
            and strat.entry_atr is not None
        ):
            trough = strat.trough_price if strat.trough_price is not None else price_val
            short_stop = min(
                strat.entry_price + strat.entry_atr * strat.atr_stop_short,
                trough * (1 + strat.trailing_stop_pct),
            )
        else:
            short_stop = np.nan

        long_stops.append(long_stop)
        short_stops.append(short_stop)
        confidences.append(strat.last_confidence)

    return pd.DataFrame(
        {
            "baseline_signal": signals,
            "long_stop": long_stops,
            "short_stop": short_stops,
            "baseline_confidence": confidences,
        },
        index=working.index,
    )


def test_vectorized_baseline_matches_and_is_faster():
    rng = np.random.default_rng(42)
    n = 3000
    base_price = rng.normal(0, 0.5, size=n).cumsum() + 100
    close = pd.Series(base_price, name="Close")
    high = close + rng.uniform(0.01, 0.5, size=n)
    low = close - rng.uniform(0.01, 0.5, size=n)
    df = pd.DataFrame({"Close": close, "High": high, "Low": low})

    params = dict(
        short_window=5,
        long_window=20,
        rsi_window=14,
        atr_window=14,
        atr_stop_long=3.0,
        atr_stop_short=3.0,
        trailing_stop_pct=0.01,
        trailing_take_profit_pct=0.02,
    )

    _sequential_reference(df.copy(), params)
    seq_start = time.perf_counter()
    sequential = _sequential_reference(df.copy(), params)
    sequential_time = time.perf_counter() - seq_start

    baseline_signal.compute(df.copy(), **params)
    vec_start = time.perf_counter()
    vectorized = baseline_signal.compute(df.copy(), **params)
    vectorized_time = time.perf_counter() - vec_start

    np.testing.assert_allclose(
        vectorized["baseline_signal"],
        sequential["baseline_signal"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        vectorized["long_stop"],
        sequential["long_stop"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        vectorized["short_stop"],
        sequential["short_stop"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        vectorized["baseline_confidence"],
        sequential["baseline_confidence"],
        equal_nan=True,
    )

    # Vectorized path should provide a meaningful speed-up
    assert vectorized_time <= sequential_time * 0.98
