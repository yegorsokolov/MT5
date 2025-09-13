# Baseline Moving-Average Strategy

The **baseline moving-average (MA) strategy** generates buy and sell orders when a
short-term moving average crosses a long-term moving average.  A configurable
``min_diff`` parameter allows ignoring small, potentially noisy differences
between the averages before triggering a trade.

```{eval-rst}
The strategy relies on :func:`indicators.common.sma` and
:func:`indicators.common.ema` for its moving-average calculations.
```

## Setup

```python
from src.strategy.registry import get_strategy
from src.strategy.executor import StrategyExecutor
from src.modes import Mode

strategy = get_strategy("baseline_ma", short_window=5, long_window=20, min_diff=0.1)
executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strategy)
```

## Indicator Usage

```python
from indicators import common as ind

prices = [1, 2, 3, 4, 5]
short_ma = ind.sma(prices, period=5)
long_ma = ind.ema(prices, period=20)
```

## Expected Behavior

- **Buy** when the short window average crosses above the long window average by
  more than ``min_diff``.
- **Sell** when the short window average crosses below the long window average by
  more than ``min_diff``.
- No order is generated when no crossover occurs.

Configure `short_window`, `long_window`, and `min_diff` to suit your market and timeframe.

## Higher Timeframe Confirmation

The strategy can optionally align entries with a broader trend by
supplying higher timeframe indicators to the ``update`` method via the
``htf_ma`` and ``htf_rsi`` parameters.  Long trades require the current
price to be above ``htf_ma`` and ``htf_rsi`` to exceed 50 while short
trades require price below ``htf_ma`` and ``htf_rsi`` below 50.  These
indicators can be derived from minute data using
``features.multi_timeframe.compute`` which resamples to intervals such as
1H or 4H and calculates moving averages and RSI.

## Kalman Moving Average Filter

For noisier markets the strategy can be combined with a Kalman-filter
based moving average (``kma``) by enabling the ``kalman_ma`` feature.
The filter smooths the closing price using two hyperparameters:

- ``process_noise`` (``Q``): variance of the process noise. Higher
  values let the filter react quicker but introduce more noise.
- ``measurement_noise`` (``R``): variance of the observation noise.
  Increasing this value produces a smoother, but more lagging, average.

When ``kalman_ma`` is active the strategy's ``update`` method accepts a
``kma_cross`` argument which signals price/KMA crossovers (``1`` for
upward, ``-1`` for downward).  New trades are only allowed when the
``kma_cross`` direction matches the moving-average signal.
