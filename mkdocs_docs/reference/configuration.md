# Configuration Models

Runtime configuration is validated by :class:`config_models.AppConfig` and its
nested models.  A configuration file can be loaded via ``utils.load_config`` and
provides access to strongly typed sections for training, features, strategies
and auxiliary services.

Key highlights:

* Legacy flat configuration files are automatically coerced into structured
  sections.
* Service commands must be declared as argument lists, protecting against
  accidental shell injection.
* ``FeaturesConfig`` deduplicates feature entries to avoid repeated execution.

## Training configuration

``TrainingConfig`` exposes strongly typed batch sizing parameters that are
validated at load time.  Configurations can now specify training batch sizes in
the dedicated section rather than relying on loosely typed top-level keys::

    training:
      model_type: lgbm
      batch_size: 128        # null lets the framework auto-tune
      eval_batch_size: 64    # optional override for validation loaders
      min_batch_size: 16     # safeguard for the auto backoff loop
      online_batch_size: 200 # controls resume-online chunking
      n_jobs: 4              # cap tree-based estimators when on shared hosts

These values are available via ``cfg.training`` as attributes and via
``cfg.get("batch_size")`` for compatibility with existing helpers.

```python
from utils import load_config
cfg = load_config("config.yaml")
print(cfg.strategy.symbols)
```

See ``docs/api/config.md`` in the Sphinx documentation for the complete API.
