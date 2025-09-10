import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure repo root on path and dynamically load the baseline_signal feature
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
spec = importlib.util.spec_from_file_location(
    "baseline_signal", ROOT / "features" / "baseline_signal.py"
)
baseline_signal = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline_signal)

from indicators import sma, rsi, atr, bollinger
from strategies.baseline import BaselineStrategy


def test_baseline_feature_matches_strategy():
    closes = [1.0, 2.0, 3.0, 2.0, 1.5, 1.6, 1.7]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    df = pd.DataFrame({"Close": closes, "High": highs, "Low": lows})

    params = dict(
        short_window=2,
        long_window=3,
        rsi_window=3,
        atr_window=2,
        atr_stop_long=1.0,
        atr_stop_short=1.0,
        trailing_stop_pct=0.05,
        trailing_take_profit_pct=0.05,
    )

    df["short_ma"] = sma(df["Close"], params["short_window"])
    ma, df["boll_upper"], df["boll_lower"] = bollinger(
        df["Close"], params["long_window"]
    )
    df["long_ma"] = ma
    df["rsi"] = rsi(df["Close"], params["rsi_window"])
    df["atr"] = atr(df["High"], df["Low"], df["Close"], params["atr_window"])

    result = baseline_signal.compute(df, **params)

    strat = BaselineStrategy(**params)
    exp_sig = []
    exp_long = []
    exp_short = []
    for row in df.itertuples():
        sig = strat.update(
            row.Close,
            high=row.High,
            low=row.Low,
            short_ma=row.short_ma if not np.isnan(row.short_ma) else None,
            long_ma=row.long_ma if not np.isnan(row.long_ma) else None,
            rsi=row.rsi if not np.isnan(row.rsi) else None,
            atr_val=row.atr if not np.isnan(row.atr) else None,
            boll_upper=row.boll_upper if not np.isnan(row.boll_upper) else None,
            boll_lower=row.boll_lower if not np.isnan(row.boll_lower) else None,
        )
        exp_sig.append(sig)
        price = row.Close
        long_stop = np.nan
        short_stop = np.nan
        if (
            strat.position == 1
            and strat.entry_price is not None
            and strat.entry_atr is not None
        ):
            peak = strat.peak_price if strat.peak_price is not None else price
            long_stop = max(
                strat.entry_price - strat.entry_atr * strat.atr_stop_long,
                peak * (1 - strat.trailing_stop_pct),
            )
        if (
            strat.position == -1
            and strat.entry_price is not None
            and strat.entry_atr is not None
        ):
            trough = strat.trough_price if strat.trough_price is not None else price
            short_stop = min(
                strat.entry_price + strat.entry_atr * strat.atr_stop_short,
                trough * (1 + strat.trailing_stop_pct),
            )
        exp_long.append(long_stop)
        exp_short.append(short_stop)

    assert result["baseline_signal"].tolist() == exp_sig
    assert np.allclose(result["baseline_long_stop"].tolist(), exp_long, equal_nan=True)
    assert np.allclose(
        result["baseline_short_stop"].tolist(), exp_short, equal_nan=True
    )
