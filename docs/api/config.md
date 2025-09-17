# Configuration Schema

The canonical configuration schema is provided by
:class:`config_models.AppConfig`.  The
:func:`config_schema.iter_config_fields` helper exposes the flattened field
metadata used by the documentation and dashboard tooling.

```{autoclass} config_models.AppConfig
:members:
:undoc-members:
:show-inheritance:
```

```{autofunction} config_schema.iter_config_fields
```

## Nested Configuration Models

The root schema composes several nested models that encapsulate specific
domains of the trading system.

```{autoclass} config_models.TrainingConfig
:members:
:show-inheritance:
```

```{autoclass} config_models.FeaturesConfig
:members:
:show-inheritance:
```

```{autoclass} config_models.StrategyConfig
:members:
:show-inheritance:
```

```{autoclass} config_models.ServicesConfig
:members:
:show-inheritance:
```
