import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from indicators import atr, bollinger, ema, rsi, sma, macd, stochastic


def _old_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _old_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _old_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def _old_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _old_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d


def test_rsi_parity():
    prices = pd.Series(np.linspace(1, 30, 30))
    expected = _old_rsi(prices, 14)
    result = rsi(prices, 14)
    pd.testing.assert_series_equal(result, expected)


def test_atr_parity():
    high = pd.Series(np.arange(1, 31) + 1)
    low = pd.Series(np.arange(1, 31) - 1)
    close = pd.Series(np.arange(1, 31))
    expected = _old_atr(high, low, close, 14)
    result = atr(high, low, close, 14)
    pd.testing.assert_series_equal(result, expected)


def test_bollinger_parity():
    prices = pd.Series(np.random.RandomState(0).randn(50).cumsum())
    exp_ma, exp_upper, exp_lower = _old_bollinger(prices, 20)
    ma, upper, lower = bollinger(prices, 20)
    pd.testing.assert_series_equal(ma, exp_ma)
    pd.testing.assert_series_equal(upper, exp_upper)
    pd.testing.assert_series_equal(lower, exp_lower)


def test_sma_parity():
    prices = pd.Series(np.arange(1, 21))
    expected = prices.rolling(5).mean()
    result = sma(prices, 5)
    pd.testing.assert_series_equal(result, expected)


def test_ema_parity():
    prices = pd.Series(np.arange(1, 21))
    expected = prices.ewm(span=5, adjust=False).mean()
    result = ema(prices, 5)
    pd.testing.assert_series_equal(result, expected)


def test_macd_parity():
    prices = pd.Series(np.arange(1, 101, dtype=float))
    exp_macd, exp_signal, exp_hist = _old_macd(prices)
    macd_line, signal_line, hist = macd(prices)
    pd.testing.assert_series_equal(macd_line, exp_macd)
    pd.testing.assert_series_equal(signal_line, exp_signal)
    pd.testing.assert_series_equal(hist, exp_hist)


def test_stochastic_parity():
    high = pd.Series(np.arange(1, 31) + 1, dtype=float)
    low = pd.Series(np.arange(1, 31) - 1, dtype=float)
    close = pd.Series(np.arange(1, 31), dtype=float)
    exp_k, exp_d = _old_stochastic(high, low, close)
    k, d = stochastic(high, low, close)
    pd.testing.assert_series_equal(k, exp_k)
    pd.testing.assert_series_equal(d, exp_d)
