from __future__ import annotations


class PluginStrategy:
    """Minimal strategy example that does nothing."""

    def update(self, data):
        del data  # unused

    def generate_order(self):
        return None


def register(register_strategy):
    """Entry point hook used by MT5 to register this strategy."""

    register_strategy("plugin_strategy", PluginStrategy)
