# Strategy Benchmarking

The project provides a lightweight harness for evaluating strategies across
multiple datasets and risk profiles.  Use
`evaluation.strategy_benchmark.run_benchmark` to execute backtests and compute
aggregate metrics such as Sharpe ratio, Conditional Value at Risk (CVaR) and
turnover.

```python
from evaluation.strategy_benchmark import run_benchmark

results = run_benchmark(my_strategy, {"sample": market_df}, [0.5, 1.0])
print(results)
```

Each row in ``results`` corresponds to a dataset and risk profile combination.
The resulting dataframe can be saved or further analyzed to compare strategies.
