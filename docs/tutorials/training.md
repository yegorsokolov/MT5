# Training Tutorial

This tutorial demonstrates how to run a simple training loop.

```bash
python -m mt5.train --config config.yaml
```

## Validating Results with Property Tests

Before deploying a strategy we rely on Hypothesis-powered tests located under
``tests/property``.  They assert invariants such as:

* Backtest cash balances never becoming negative when using capped leverage.
* Position sizes respecting the configured maximum exposure.
* Purged time-series splits excluding embargoed samples and overlapping groups.

Running ``pytest tests/property`` is a fast way to sanity check changes to the
feature pipeline or strategy logic.

```{doctest}
>>> def train(cfg):
...     return cfg["epochs"]
>>> train({"epochs": 1})
1
```
