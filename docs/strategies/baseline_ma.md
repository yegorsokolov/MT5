# Baseline Moving-Average Strategy

The **baseline moving-average (MA) strategy** generates buy and sell orders when a
short-term moving average crosses a long-term moving average.

## Setup

```python
from src.strategy.registry import get_strategy
from src.strategy.executor import StrategyExecutor
from src.modes import Mode

strategy = get_strategy("baseline_ma", short_window=5, long_window=20)
executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strategy)
```

## Expected Behavior

- **Buy** when the short window average crosses above the long window average.
- **Sell** when the short window average crosses below the long window average.
- No order is generated when no crossover occurs.

Configure `short_window` and `long_window` to suit your market and timeframe.
