# Purged Cross-Validation Invariants

Property-based tests verify that the ``PurgedTimeSeriesSplit`` utility
enforces two critical safety guarantees:

- **Embargo compliance** – training samples occur strictly before the
  validation fold and are separated from it by the configured embargo
  window.
- **Group exclusion** – no training sample shares a group with any
  validation sample.

Together these invariants reduce the risk of information leakage in
time-series model evaluation.

