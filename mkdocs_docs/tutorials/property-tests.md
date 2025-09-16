# Property-Based Testing Workflow

Hypothesis-driven tests exercise the backtest and cross-validation subsystems
using randomly generated market scenarios.  To run them locally:

```bash
pip install -r requirements.txt hypothesis
pytest tests/property -q
```

The suite verifies that:

1. Cash balances remain non-negative when respecting leverage constraints.
2. Position sizes never exceed configured limits regardless of signal noise.
3. Purged cross-validation folds exclude embargoed and overlapping group
   samples.

These guardrails catch subtle regressions before they reach integration tests
or production deployments.
