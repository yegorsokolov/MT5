# Baseline Moving-Average Strategy

The **baseline moving-average (MA) strategy** generates buy and sell orders when a
short-term moving average crosses a long-term moving average.  A configurable
``min_diff`` parameter allows ignoring small, potentially noisy differences
between the averages before triggering a trade.

## Setup

```python
from src.strategy.registry import get_strategy
from src.strategy.executor import StrategyExecutor
from src.modes import Mode

strategy = get_strategy("baseline_ma", short_window=5, long_window=20, min_diff=0.1)
executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strategy)
```

## Expected Behavior

- **Buy** when the short window average crosses above the long window average by
  more than ``min_diff``.
- **Sell** when the short window average crosses below the long window average by
  more than ``min_diff``.
- No order is generated when no crossover occurs.

Configure `short_window`, `long_window`, and `min_diff` to suit your market and timeframe.
