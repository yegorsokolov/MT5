import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

# Stub minimal utils module to avoid heavy dependency initialization
utils_stub = types.ModuleType("utils")

def _dummy_load_config():
    class _Cfg:
        strategy = types.SimpleNamespace(
            session_position_limits={}, default_position_limit=1
        )

    return _Cfg()

utils_stub.load_config = _dummy_load_config
sys.modules.setdefault("utils", utils_stub)

# Provide a lightweight ``pydantic`` shim so strategy imports succeed without the
# heavy dependency.
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

# Ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import deque

import numpy as np
import pandas as pd
import pytest

from strategies.baseline import BaselineStrategy, IndicatorBundle
from indicators import atr as calc_atr, bollinger, rsi as calc_rsi, sma
from backtesting import walk_forward


def test_trailing_stop_exit_long():
    strategy = BaselineStrategy(
        short_window=2,
        long_window=3,
        atr_window=2,
        atr_stop_long=1.0,
        trailing_stop_pct=0.05,
        trailing_take_profit_pct=0.05,
    )
    closes = [1, 2, 3, 2.9, 2.5]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signals = [
        strategy.update(c, IndicatorBundle(high=h, low=l))
        for c, h, l in zip(closes, highs, lows)
    ]
    assert signals[2] == 1  # Buy on crossover
    assert signals[-1] == -1  # Exit via trailing stop


def test_trailing_take_profit_exit():
    strategy = BaselineStrategy(
        short_window=2,
        long_window=3,
        atr_window=2,
        atr_stop_long=0.1,
        trailing_stop_pct=0.05,
        trailing_take_profit_pct=0.05,
    )
    closes = [1, 2, 3, 3.5, 3.6, 3.3]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signals = [
        strategy.update(c, IndicatorBundle(high=h, low=l))
        for c, h, l in zip(closes, highs, lows)
    ]
    assert signals[2] == 1  # Buy on crossover
    assert signals[-1] == -1  # Trailing take-profit triggers exit


def test_external_indicators_match_internal():
    strat_internal = BaselineStrategy(
        short_window=2, long_window=3, rsi_window=3, atr_window=2
    )
    strat_external = BaselineStrategy(
        short_window=2, long_window=3, rsi_window=3, atr_window=2
    )

    closes = [1, 2, 3, 2, 1.5]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]

    # Containers for manual indicator calculations
    short_q = deque(maxlen=2)
    long_q = deque(maxlen=3)
    highs_q = deque(maxlen=3)
    lows_q = deque(maxlen=3)
    closes_q = deque(maxlen=3)

    signals_internal = []
    signals_external = []

    for c, h, l in zip(closes, highs, lows):
        signals_internal.append(
            strat_internal.update(c, IndicatorBundle(high=h, low=l))
        )

        short_q.append(c)
        long_q.append(c)
        highs_q.append(h)
        lows_q.append(l)
        closes_q.append(c)

        if len(long_q) >= 3 and len(closes_q) >= 3:
            short_ma = sma(short_q, 2)
            long_ma, upper, lower = bollinger(long_q, 3)
            rsi_val = calc_rsi(long_q, 3)
            atr_val = calc_atr(highs_q, lows_q, closes_q, 2)
        else:
            short_ma = long_ma = rsi_val = atr_val = upper = lower = None

        ind = IndicatorBundle(
            high=h,
            low=l,
            short_ma=short_ma,
            long_ma=long_ma,
            rsi=rsi_val,
            atr_val=atr_val,
            boll_upper=upper,
            boll_lower=lower,
        )
        signals_external.append(strat_external.update(c, ind))

    assert signals_internal == signals_external


def test_update_accepts_evolved_indicators():
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=2)
    ind = IndicatorBundle(evolved={"custom": 1.23})
    # Should run without raising and return a numeric signal
    sig = strat.update(1.0, ind)
    assert isinstance(sig, float)


def test_indicator_bundle_vector_length_with_evolved():
    series = pd.Series([1.0, 2.0, 3.0])
    bundle = IndicatorBundle(short_ma=series, evolved={"foo": np.array([0.1, 0.2, 0.3])})
    assert bundle.vector_length() == 3

    mismatch = IndicatorBundle(short_ma=series, evolved={"foo": np.array([0.1, 0.2])})
    with pytest.raises(ValueError):
        mismatch.vector_length()


def test_batch_update_rejects_length_mismatch():
    price = pd.Series([1.0, 1.1, 1.2, 1.3])
    bundle = IndicatorBundle(
        high=price,
        low=price,
        evolved={"foo": np.array([1.0, 2.0, 3.0])},
    )
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=2)
    with pytest.raises(ValueError):
        strat.batch_update(price, bundle)


def test_batch_update_vectorized_evolved_features_end_to_end(tmp_path):
    rng = np.random.default_rng(7)
    n = 80
    t = np.linspace(0, 6 * np.pi, n)
    base = 100 + np.sin(t) + 0.05 * t
    close = pd.Series(base + rng.normal(scale=0.05, size=n), name="Close")
    high = close + rng.uniform(0.05, 0.2, size=n)
    low = close - rng.uniform(0.05, 0.2, size=n)
    df = pd.DataFrame({"Close": close, "High": high, "Low": low})

    formulas = [
        {
            "name": "gate",
            "formula": "(df['Close'].rolling(4).mean().fillna(df['Close']) > df['Close'].rolling(10).mean().fillna(df['Close'])).astype(float)",
        },
        {"name": "trend_strength", "formula": "np.cos(np.linspace(0, np.pi, len(df)))"},
    ]
    formula_path = tmp_path / "evolved_formulas.json"
    formula_path.write_text(json.dumps(formulas))

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "baseline_evolved_indicators",
        root / "features" / "evolved_indicators.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    df_evolved = module.compute(df.copy(), path=formula_path)

    bundle = IndicatorBundle(
        high=df_evolved["High"],
        low=df_evolved["Low"],
        evolved={
            "gate": df_evolved["gate"].astype(float),
            "trend_strength": df_evolved["trend_strength"].astype(float),
        },
    )
    assert bundle.vector_length() == len(df_evolved)

    params = dict(
        short_window=3,
        long_window=8,
        rsi_window=6,
        atr_window=7,
        atr_stop_long=1.5,
        atr_stop_short=1.5,
        trailing_stop_pct=0.02,
        trailing_take_profit_pct=0.03,
    )
    strat_vectorized = BaselineStrategy(**params)
    vec_signal, vec_long, vec_short = strat_vectorized.batch_update(
        df_evolved["Close"], bundle
    )

    strat_sequential = BaselineStrategy(**params)
    seq_signal: list[float] = []
    seq_long: list[float] = []
    seq_short: list[float] = []
    for row in df_evolved.itertuples():
        ind = IndicatorBundle(
            high=row.High,
            low=row.Low,
            evolved={
                "gate": float(row.gate),
                "trend_strength": float(row.trend_strength),
            },
        )
        sig = strat_sequential.update(row.Close, ind)
        seq_signal.append(sig)

        if (
            strat_sequential.position == 1
            and strat_sequential.entry_price is not None
            and strat_sequential.entry_atr is not None
        ):
            peak = (
                strat_sequential.peak_price
                if strat_sequential.peak_price is not None
                else row.Close
            )
            long_stop = max(
                strat_sequential.entry_price
                - strat_sequential.entry_atr * strat_sequential.atr_stop_long,
                peak * (1 - strat_sequential.trailing_stop_pct),
            )
        else:
            long_stop = np.nan

        if (
            strat_sequential.position == -1
            and strat_sequential.entry_price is not None
            and strat_sequential.entry_atr is not None
        ):
            trough = (
                strat_sequential.trough_price
                if strat_sequential.trough_price is not None
                else row.Close
            )
            short_stop = min(
                strat_sequential.entry_price
                + strat_sequential.entry_atr * strat_sequential.atr_stop_short,
                trough * (1 + strat_sequential.trailing_stop_pct),
            )
        else:
            short_stop = np.nan

        seq_long.append(long_stop)
        seq_short.append(short_stop)

    np.testing.assert_allclose(vec_signal, np.asarray(seq_signal), equal_nan=True)
    np.testing.assert_allclose(vec_long, np.asarray(seq_long), equal_nan=True)
    np.testing.assert_allclose(vec_short, np.asarray(seq_short), equal_nan=True)

    gating = df_evolved["gate"].astype(float)
    price_returns = df_evolved["Close"].pct_change().fillna(0.0)
    vec_returns = (
        pd.Series(vec_signal, index=df_evolved.index)
        .shift(1)
        .fillna(0.0)
        * gating.shift(1).fillna(0.0)
        * price_returns
    )
    seq_returns = (
        pd.Series(seq_signal, index=df_evolved.index)
        .shift(1)
        .fillna(0.0)
        * gating.shift(1).fillna(0.0)
        * price_returns
    )
    pd.testing.assert_series_equal(vec_returns, seq_returns)

    metrics = walk_forward.aggregate_metrics(
        pd.DataFrame({"return": vec_returns}), train_size=20, val_size=10, step=10
    )
    assert set(metrics) == {"avg_sharpe", "worst_drawdown"}
    for value in metrics.values():
        assert math.isfinite(value) or math.isnan(value)
