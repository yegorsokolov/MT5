# Backtest Invariants

Property-based tests in `tests/property/test_backtest_props.py` validate two core invariants of the differentiable backtest utilities:

1. **Inventory bounds** – positions produced by `soft_position` are always confined to the range `[-1, 1]`.  This prevents the simulator from taking on exposure larger than one unit in either direction.
2. **PnL monotonicity with zero slippage** – when execution is free of slippage, improving the price path (increasing every increment) cannot reduce cumulative PnL for non‑negative positions.

These invariants help ensure the backtesting components behave sensibly under random price paths and strategies.
