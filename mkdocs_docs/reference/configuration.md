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

```python
from utils import load_config
cfg = load_config("config.yaml")
print(cfg.strategy.symbols)
```

See ``docs/api/config.md`` in the Sphinx documentation for the complete API.
