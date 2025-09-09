# Strategy Approval Process

Strategies begin in an **experimental** state.  Each strategy has an entry in
`strategies/metadata.json` tracking its evaluation status, performance metrics
and whether it is approved for live trading.

## Benchmarks

A strategy is considered for approval after backtesting achieves:

* Sharpe ratio above **1.0**
* No failing risk checks

The metrics are produced by `src/evaluation/backtest_runner.py` which runs the
standard backtest and updates `metadata.json` with the results.  Approval is set
automatically based on the Sharpe ratio threshold.

## Promotion to Approved

1. Run the backtest runner for the strategy:

   ```bash
   python -m src.evaluation.backtest_runner STRATEGY_NAME
   ```

2. Review the updated entry in `strategies/metadata.json`.
3. If benchmarks are met the strategy's `status` changes to `approved` and the
   flag `approved` is set to `true`.
4. Approved strategies may then be executed in live trading.  The
   `StrategyExecutor` checks the metadata file before placing any live orders.

Strategies failing to meet the benchmarks remain experimental and are blocked
from live execution.
