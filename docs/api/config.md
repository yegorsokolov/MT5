# Configuration Schema

```{autoclass} config_schema.ConfigSchema
:members:
:undoc-members:
:show-inheritance:
:exclude-members: check_symbols
```

## Runtime Configuration Models

The declarative schema above is complemented by a set of runtime configuration
models built on :mod:`pydantic`.  They provide strict validation, convenient
defaults, and helper methods used throughout the trading stack.

```{autoclass} config_models.AppConfig
:members:
:show-inheritance:
```

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
