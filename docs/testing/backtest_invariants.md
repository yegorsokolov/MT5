# Backtest Invariants

Property-based tests exercise random price paths and trading strategies to ensure core backtest invariants hold:

- **No negative cash** – trades never reduce the cash balance below zero.
- **Position limits** – positions are clipped to a maximum absolute size of one unit.

These checks provide a safety net for future refactors of the simulation engine.
