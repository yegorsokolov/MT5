# Indicators

The indicator toolbox exposes a curated collection of reusable statistical
features that power both backtests and live trading deployments.  Frequently
used helpers are summarised below.

| Indicator | Description |
| --- | --- |
| ``sma`` | Rolling simple moving average with graceful handling of Python sequences and pandas objects. |
| ``ema`` | Exponentially weighted moving average suitable for faster reacting trend signals. |
| ``rsi`` | Momentum oscillator bounded between 0 and 100 exposing overbought and oversold regimes. |
| ``bollinger`` | Upper and lower bands around a moving average capturing price dispersion. |
| ``atr`` | Average true range highlighting market volatility using high/low/close information. |

```{automodule} indicators.common
:members:
:undoc-members:
```
