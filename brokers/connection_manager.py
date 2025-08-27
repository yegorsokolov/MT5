import asyncio
import logging
import time
from typing import Any, List, Optional

from metrics import BROKER_FAILURES, BROKER_LATENCY_MS

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Maintain connections to multiple broker backends.

    The manager keeps a list of broker modules/objects. Each broker is expected
    to expose an ``initialize`` function and relevant API methods used by the
    application (e.g. ``copy_ticks_from``).

    When the active broker fails, :meth:`failover` will attempt to connect to a
    backup broker. The currently active broker can be retrieved via
    :func:`get_active_broker`.
    """

    def __init__(self, brokers: List[Any]):
        if not brokers:
            raise ValueError("At least one broker must be provided")
        self._brokers = brokers
        self._active_index: Optional[int] = None
        self._failure_counts = {i: 0 for i in range(len(brokers))}
        self._connect_first_available()

    def _connect_first_available(self) -> None:
        """Initialize the first available broker."""
        for i, broker in enumerate(self._brokers):
            try:
                if broker.initialize():
                    self._active_index = i
                    logger.info("Connected to broker %s", getattr(broker, "__name__", broker))
                    return
            except Exception:
                logger.exception("Failed to initialize broker %s", broker)
        raise RuntimeError("No broker connections could be established")

    def get_active_broker(self) -> Any:
        if self._active_index is None:
            raise RuntimeError("Connection manager not initialized")
        return self._brokers[self._active_index]

    def failover(self) -> bool:
        """Attempt to switch to a backup broker.

        Returns ``True`` if a new broker was activated, ``False`` otherwise.
        """
        if self._active_index is None:
            return False
        num = len(self._brokers)
        for offset in range(1, num + 1):
            idx = (self._active_index + offset) % num
            broker = self._brokers[idx]
            try:
                if broker.initialize():
                    self._active_index = idx
                    logger.warning(
                        "Failover: switched to broker %s", getattr(broker, "__name__", broker)
                    )
                    return True
            except Exception:
                logger.exception("Failed to initialize broker %s during failover", broker)
        logger.error("All brokers failed during failover")
        return False

    async def watchdog(
        self,
        interval: float = 5.0,
        timeout: float = 1.0,
        latency_threshold_ms: float = 1000.0,
        failure_threshold: int = 3,
    ) -> None:
        """Periodically ping brokers and trigger failover on degradation."""

        try:
            while True:
                for i, broker in enumerate(self._brokers):
                    name = getattr(broker, "__name__", broker.__class__.__name__)
                    start = time.perf_counter()
                    try:
                        if hasattr(broker, "ping"):
                            func = broker.ping
                        else:
                            func = broker.initialize
                        if asyncio.iscoroutinefunction(func):
                            await asyncio.wait_for(func(), timeout=timeout)
                        else:
                            await asyncio.wait_for(asyncio.to_thread(func), timeout=timeout)
                        latency = (time.perf_counter() - start) * 1000
                        BROKER_LATENCY_MS.labels(broker=name).set(latency)
                        if latency > latency_threshold_ms:
                            self._failure_counts[i] += 1
                            BROKER_FAILURES.labels(broker=name).inc()
                        else:
                            self._failure_counts[i] = 0
                    except Exception:
                        BROKER_FAILURES.labels(broker=name).inc()
                        self._failure_counts[i] += 1
                    if i == self._active_index and self._failure_counts[i] >= failure_threshold:
                        self.failover()
                        self._failure_counts[i] = 0
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

_manager: Optional[ConnectionManager] = None


def init(brokers: List[Any]) -> None:
    """Initialise the global connection manager with ``brokers``."""
    global _manager
    _manager = ConnectionManager(brokers)


def get_active_broker() -> Any:
    if _manager is None:
        raise RuntimeError("Connection manager not initialized")
    return _manager.get_active_broker()


def failover() -> bool:
    if _manager is None:
        return False
    return _manager.failover()


async def watchdog(**kwargs) -> None:
    if _manager is None:
        return
    await _manager.watchdog(**kwargs)
