# Extending MT5 with Plugins

MT5 exposes plugin hooks for adding new indicators and trading strategies without
modifying the core repository.  Plugins are standard Python packages that
declare entry points so MT5 can automatically discover and register them at
runtime.

## Creating a Plugin

1. Create a Python package with a `pyproject.toml` file.
2. Add entry points under the `mt5.indicators` and/or `mt5.strategies` groups.
3. Each entry point should reference a callable named ``register`` that accepts a
   registration function provided by MT5.

Example `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "mt5-example-plugin"
version = "0.1.0"

[project.entry-points."mt5.indicators"]
example = "example_plugin.indicator:register"

[project.entry-points."mt5.strategies"]
example = "example_plugin.strategy:register"
```

## Registering Indicators

The entry point ``example_plugin.indicator:register`` might look like:

```python
# example_plugin/indicator.py

def compute(df):
    df["plugin_indicator"] = df["close"].pct_change().fillna(0)
    return df


def register(register_feature):
    register_feature("plugin_indicator", compute)
```

## Registering Strategies

```python
# example_plugin/strategy.py

class PluginStrategy:
    def __init__(self):
        self._last_price = None

    def update(self, data):
        self._last_price = data.get("close")

    def generate_order(self):
        return None


def register(register_strategy):
    register_strategy("plugin_strategy", PluginStrategy)
```

See [`plugins/example_plugin`](../plugins/example_plugin/) for a complete
working example.
