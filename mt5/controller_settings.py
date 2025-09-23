from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Callable, Dict, Optional, Set

try:  # pragma: no cover - optional during some tests
    from utils.resource_monitor import monitor as _monitor
except Exception:  # pragma: no cover - lightweight environments
    _monitor = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControllerSettings:
    """Runtime tuning parameters for the local controller."""

    max_rss_mb: Optional[float]
    max_cpu_pct: Optional[float]
    watchdog_usec: int
    bot_backoff_base: float
    bot_backoff_max: float
    bot_backoff_reset: float
    bot_max_crashes: int
    bot_crash_window: float


_LOCK = RLock()
_SETTINGS: ControllerSettings
_SUBSCRIBERS: Set[Callable[[ControllerSettings], None]] = set()


def _read_optional_positive_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        LOGGER.warning("Ignoring invalid %s override: %s", name, raw)
        return default
    return value if value > 0 else None


def _read_non_negative_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        LOGGER.warning("Ignoring invalid %s override: %s", name, raw)
        return default
    return value if value >= 0 else default


def _read_non_negative_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        LOGGER.warning("Ignoring invalid %s override: %s", name, raw)
        return default
    return max(value, minimum)


def _default_max_rss_mb() -> Optional[float]:
    caps = getattr(_monitor, "capabilities", None)
    if not caps:
        return None
    memory_gb = getattr(caps, "memory_gb", 0.0) or 0.0
    if memory_gb <= 0:
        return None
    return round(memory_gb * 1024 * 0.8, 2)


def _default_max_cpu_pct() -> Optional[float]:
    caps = getattr(_monitor, "capabilities", None)
    if not caps:
        return None
    cpus = getattr(caps, "cpus", 0) or 0
    if cpus <= 0:
        return None
    return round(cpus * 100 * 0.8, 2)


def _coerce_optional_positive(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def _coerce_non_negative_float(value: Any, *, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return num if num >= 0 else default


def _coerce_non_negative_int(value: Any, *, default: int = 0, minimum: int = 0) -> int:
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return max(default, minimum)
    return max(num, minimum)


def _build_settings(data: Dict[str, Any]) -> ControllerSettings:
    max_rss = _coerce_optional_positive(data.get("max_rss_mb"))
    max_cpu = _coerce_optional_positive(data.get("max_cpu_pct"))
    watchdog = _coerce_non_negative_int(data.get("watchdog_usec"), default=0, minimum=0)
    base = _coerce_non_negative_float(data.get("bot_backoff_base"), default=1.0)
    max_backoff = _coerce_non_negative_float(data.get("bot_backoff_max"), default=60.0)
    if max_backoff < base:
        max_backoff = base
    reset = _coerce_non_negative_float(data.get("bot_backoff_reset"), default=300.0)
    max_crashes = _coerce_non_negative_int(data.get("bot_max_crashes"), default=5, minimum=1)
    crash_window = _coerce_non_negative_float(data.get("bot_crash_window"), default=600.0)
    crash_window = max(crash_window, base)
    return ControllerSettings(
        max_rss_mb=max_rss,
        max_cpu_pct=max_cpu,
        watchdog_usec=watchdog,
        bot_backoff_base=base,
        bot_backoff_max=max_backoff,
        bot_backoff_reset=reset,
        bot_max_crashes=max_crashes,
        bot_crash_window=crash_window,
    )


def _initial_settings() -> ControllerSettings:
    data: Dict[str, Any] = {
        "max_rss_mb": _read_optional_positive_float("MAX_RSS_MB", _default_max_rss_mb()),
        "max_cpu_pct": _read_optional_positive_float("MAX_CPU_PCT", _default_max_cpu_pct()),
        "watchdog_usec": _read_non_negative_int("WATCHDOG_USEC", 0, minimum=0),
        "bot_backoff_base": _read_non_negative_float("BOT_BACKOFF_BASE", 1.0),
        "bot_backoff_max": _read_non_negative_float("BOT_BACKOFF_MAX", 60.0),
        "bot_backoff_reset": _read_non_negative_float("BOT_BACKOFF_RESET", 300.0),
        "bot_max_crashes": _read_non_negative_int("BOT_MAX_CRASHES", 5, minimum=1),
        "bot_crash_window": _read_non_negative_float("BOT_CRASH_WINDOW", 600.0),
    }
    return _build_settings(data)


def _notify(settings: ControllerSettings) -> None:
    for callback in list(_SUBSCRIBERS):
        try:
            callback(settings)
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Controller settings subscriber failed")


def get_controller_settings() -> ControllerSettings:
    """Return the current controller settings."""

    with _LOCK:
        return _SETTINGS


def update_controller_settings(*, only_if_missing: bool = False, **overrides: Any) -> ControllerSettings:
    """Update controller settings and notify subscribers.

    Parameters
    ----------
    only_if_missing:
        When ``True``, only apply overrides for fields that are currently unset
        (``None``). This is primarily used by auto-tuning helpers to avoid
        clobbering manual overrides.
    overrides:
        Mapping of field names to new values.
    """

    if not overrides:
        return get_controller_settings()

    global _SETTINGS
    with _LOCK:
        current = asdict(_SETTINGS)
        for field, value in overrides.items():
            if field not in current:
                raise KeyError(f"Unknown controller setting: {field}")
            if only_if_missing and current[field] is not None:
                continue
            current[field] = value
        new_settings = _build_settings(current)
        _SETTINGS = new_settings
    _notify(new_settings)
    return new_settings


def subscribe_controller_settings(
    callback: Callable[[ControllerSettings], None]
) -> Callable[[], None]:
    """Register ``callback`` to run whenever settings change.

    The callback is invoked immediately with the current settings. The returned
    callable removes the subscription when invoked.
    """

    with _LOCK:
        _SUBSCRIBERS.add(callback)
        settings = _SETTINGS
    try:
        callback(settings)
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("Controller settings subscriber failed during registration")

    def _unsubscribe() -> None:
        with _LOCK:
            _SUBSCRIBERS.discard(callback)

    return _unsubscribe


def auto_tune_controller_settings(
    monitor: Any | None = None, *, fraction: float = 0.8, force: bool = False
) -> ControllerSettings:
    """Derive sensible defaults based on the provided ``monitor``.

    ``fraction`` controls the proportion of available resources reserved for
    the watchdog thresholds. When ``force`` is ``False`` (the default), limits
    are only applied when the existing value is ``None`` so manual overrides are
    preserved.
    """

    mon = monitor or _monitor
    caps = getattr(mon, "capabilities", None)
    if not caps:
        return get_controller_settings()

    updates: Dict[str, Any] = {}
    memory_gb = getattr(caps, "memory_gb", 0.0) or 0.0
    if memory_gb > 0:
        updates["max_rss_mb"] = round(memory_gb * 1024 * fraction, 2)
    cpus = getattr(caps, "cpus", 0) or 0
    if cpus > 0:
        updates["max_cpu_pct"] = round(cpus * 100 * fraction, 2)

    if not updates:
        return get_controller_settings()

    return update_controller_settings(only_if_missing=not force, **updates)


_SETTINGS = _initial_settings()

__all__ = [
    "ControllerSettings",
    "get_controller_settings",
    "update_controller_settings",
    "subscribe_controller_settings",
    "auto_tune_controller_settings",
]
