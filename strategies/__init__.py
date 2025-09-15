"""Strategy registry with plugin discovery.

This module mirrors the feature registry by providing a lightweight
registration API that can be used by both built-in strategies and external
plugins.  Strategies register a factory callable – typically the strategy
class itself – under a human friendly name.  Consumers such as CLI tools can
then query and instantiate these strategies at runtime without a hard coded
list.  Plugins are discovered via the ``mt5.strategies`` entry-point group,
matching the behaviour of the feature registry.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable, Dict, Iterable, Protocol, TypeVar

logger = logging.getLogger(__name__)


class StrategyProtocol(Protocol):
    """Protocol describing the minimal strategy interface."""

    def generate_order(self, market_data: dict) -> object:  # pragma: no cover - protocol
        """Return an order for ``market_data``."""

    def update(self, *args, **kwargs) -> None:  # pragma: no cover - protocol
        """Update internal state after an execution event."""


StrategyT = TypeVar("StrategyT", bound=StrategyProtocol)
StrategyFactory = Callable[..., StrategyT]


@dataclass
class StrategySpec:
    """Specification for a registered strategy."""

    factory: StrategyFactory
    description: str | None = None


_REGISTRY: Dict[str, StrategySpec] = {}
_external_loaded = False
_defaults_loaded = False


def register_strategy(
    name: str,
    factory: StrategyFactory,
    *,
    description: str | None = None,
) -> None:
    """Register ``factory`` under ``name``.

    Parameters
    ----------
    name:
        Identifier used to look up the strategy.
    factory:
        Callable returning a strategy instance.  Passing a class is valid
        because classes are callables.
    description:
        Optional human readable description shown by tooling.
    """

    if not callable(factory):  # pragma: no cover - defensive programming
        raise TypeError("factory must be callable")
    _REGISTRY[name] = StrategySpec(factory=factory, description=description)


def available_strategies() -> Dict[str, StrategySpec]:
    """Return mapping of registered strategies after loading plugins."""

    _ensure_default_strategies()
    _load_external_strategies()
    return dict(_REGISTRY)


def iter_strategies() -> Iterable[tuple[str, StrategySpec]]:
    """Yield ``(name, spec)`` pairs for all registered strategies."""

    return available_strategies().items()


def create_strategy(name: str, /, **kwargs) -> StrategyT:
    """Instantiate the strategy registered as ``name``."""

    spec = available_strategies().get(name)
    if spec is None:
        raise KeyError(f"Unknown strategy: {name}")
    return spec.factory(**kwargs)


def _load_external_strategies() -> None:
    """Load strategies declared via ``mt5.strategies`` entry points."""

    global _external_loaded
    if _external_loaded:
        return
    try:
        eps = entry_points(group="mt5.strategies")
    except TypeError:  # pragma: no cover - Python < 3.10 fallback
        eps = entry_points().get("mt5.strategies", [])
    except Exception:  # pragma: no cover - metadata unavailable
        logger.debug("Failed to query strategy entry points", exc_info=True)
        eps = []
    for ep in eps:
        try:
            hook = ep.load()
            hook(register_strategy)
            logger.info("Loaded strategy plugin %s", ep.name)
        except Exception:  # pragma: no cover - plugin errors shouldn't crash
            logger.debug("Failed to load strategy plugin %s", ep.name, exc_info=True)
    _external_loaded = True


def register_builtin_strategies(register: Callable[[str, StrategyFactory], None]) -> None:
    """Entry-point hook to register built-in strategies when installed."""

    try:
        from .baseline import BaselineStrategy

        register(
            "baseline",
            BaselineStrategy,
            description="Enhanced moving-average baseline strategy",
        )
    except Exception:  # pragma: no cover - optional during docs/tests
        logger.debug("Failed to register BaselineStrategy", exc_info=True)

    try:
        from src.strategy.baseline_ma import BaselineMovingAverageStrategy

        register(
            "baseline_ma",
            BaselineMovingAverageStrategy,
            description="Simple moving-average crossover baseline",
        )
    except Exception:  # pragma: no cover - optional dependency in some envs
        logger.debug("Failed to register BaselineMovingAverageStrategy", exc_info=True)


def _ensure_default_strategies() -> None:
    global _defaults_loaded
    if _defaults_loaded or os.getenv("MT5_DOCS_BUILD"):
        return
    register_builtin_strategies(register_strategy)
    _defaults_loaded = True


_ensure_default_strategies()


__all__ = [
    "StrategyProtocol",
    "StrategySpec",
    "available_strategies",
    "create_strategy",
    "iter_strategies",
    "register_strategy",
    "register_builtin_strategies",
]
