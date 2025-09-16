# Purged Time-Series Cross-Validation

The :class:`analysis.purged_cv.PurgedTimeSeriesSplit` helper mirrors the API of
``sklearn.model_selection.TimeSeriesSplit`` while adding two safeguards that are
critical for financial datasets:

* **Embargo support** prevents training examples from peeking into labels that
  extend into the validation horizon.
* **Group awareness** removes samples sharing a symbol or regime label with the
  validation fold.  A configurable ``group_gap`` drops neighbouring groups as
  well, mirroring ``GroupTimeSeriesSplit`` semantics.

```python
from analysis.purged_cv import PurgedTimeSeriesSplit
splitter = PurgedTimeSeriesSplit(n_splits=4, embargo=5, group_gap=2)
for train_idx, val_idx in splitter.split(range(200), groups=symbol_ids):
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]
```

Property-based tests under ``tests/property/test_purged_cv.py`` assert that
embargoed ranges and duplicate groups never leak into the training indices.
