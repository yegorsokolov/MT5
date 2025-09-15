import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import numpy as np

from indicators.common import atr, bollinger, rsi, sma

# Provide a minimal utils stub before importing BaselineStrategy
utils_stub = types.ModuleType("utils")

def _dummy_load_config():
    class _Cfg:
        strategy = types.SimpleNamespace(
            session_position_limits={}, default_position_limit=1
        )

    return _Cfg()

utils_stub.load_config = _dummy_load_config
sys.modules.setdefault("utils", utils_stub)

from strategies.baseline import BaselineStrategy, IndicatorBundle


def _load_price_module():
    pkg_name = "features"
    pkg_path = Path(__file__).resolve().parents[1] / pkg_name
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.price", pkg_path / "price.py"
    )
    price_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(price_mod)
    return price_mod


def test_price_features_match_indicator_functions():
    price = _load_price_module()
    df = pd.DataFrame(
        {
            "Bid": np.linspace(100, 120, 100),
            "Ask": np.linspace(100.1, 120.1, 100),
            "Timestamp": pd.date_range("2023", periods=100, freq="T"),
        }
    )
    feats = price.compute(df)
    mid = feats["mid"]

    pd.testing.assert_series_equal(feats["ma_5"], sma(mid, 5), check_names=False)
    pd.testing.assert_series_equal(feats["ma_10"], sma(mid, 10), check_names=False)
    pd.testing.assert_series_equal(feats["rsi_14"], rsi(mid, 14), check_names=False)
    pd.testing.assert_series_equal(
        feats["atr_14"], atr(mid, mid, mid, 14), check_names=False
    )
    _, upper, lower = bollinger(mid, 20)
    pd.testing.assert_series_equal(feats["boll_upper"], upper, check_names=False)
    pd.testing.assert_series_equal(feats["boll_lower"], lower, check_names=False)


def test_baseline_strategy_with_precomputed_indicators_matches_internal():
    df = pd.DataFrame(
        {
            "Bid": np.linspace(1, 2, 100),
            "Ask": np.linspace(1.1, 2.1, 100),
            "Timestamp": pd.date_range("2023", periods=100, freq="T"),
        }
    )
    mid = (df["Bid"] + df["Ask"]) / 2
    strat_pre = BaselineStrategy(short_window=5, long_window=10)
    strat_internal = BaselineStrategy(short_window=5, long_window=10)

    sma_short = sma(mid, 5)
    sma_long = sma(mid, 10)
    rsi_series = rsi(mid, 14)
    atr_series = atr(mid, mid, mid, 14)
    _, ub, lb = bollinger(mid, 10)

    signals_pre = []
    signals_internal = []
    for i, price in enumerate(mid):
        ind = IndicatorBundle(
            short_ma=None if pd.isna(sma_short.iloc[i]) else float(sma_short.iloc[i]),
            long_ma=None if pd.isna(sma_long.iloc[i]) else float(sma_long.iloc[i]),
            rsi=None if pd.isna(rsi_series.iloc[i]) else float(rsi_series.iloc[i]),
            atr_val=None
            if (i < strat_pre.atr_window or pd.isna(atr_series.iloc[i]))
            else float(atr_series.iloc[i]),
            boll_upper=None if pd.isna(ub.iloc[i]) else float(ub.iloc[i]),
            boll_lower=None if pd.isna(lb.iloc[i]) else float(lb.iloc[i]),
        )
        signals_pre.append(strat_pre.update(float(price), ind))
        signals_internal.append(strat_internal.update(float(price)))

    assert signals_pre == signals_internal
