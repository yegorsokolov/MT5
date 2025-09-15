from __future__ import annotations


class PluginStrategy:
    """Minimal strategy example used by tests."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)

    def generate_order(self, tick):
        price = float(tick.get("price", 0.0))
        if price > self.threshold:
            return {"quantity": 1}
        if price < -self.threshold:
            return {"quantity": -1}
        return {"quantity": 0}

    def update(self, *_, **__):  # pragma: no cover - simple no-op
        return None


def register(register_strategy):
    """Entry point hook used by MT5 to register this strategy."""

    register_strategy("plugin_strategy", PluginStrategy, description="Example plugin strategy")
