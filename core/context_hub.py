"""Lightweight context bus for cross-component communication.

The trading stack contains multiple loosely coupled services – the risk
manager tracks drawdowns, the orchestrator supervises resources and model
deployment while the AI control plane reasons about router/risk state.
Historically these components only exchanged information indirectly via log
files or ad-hoc function calls, which made it difficult for any subsystem to
reason about the "big picture" view of the platform.

This module provides a very small publish/subscribe style hub that stores the
latest state for each component and exposes it through thread-safe snapshot
queries or asynchronous subscriptions.  Producers call :func:`update` with a
component name and payload.  Consumers can either poll
:func:`context_hub.snapshot` for the merged state or subscribe to a queue that
receives updates whenever any component publishes a new payload.

The design deliberately avoids heavy dependencies so it can be used from data
loading code, the orchestrator event loop or background worker threads.  All
payloads are normal dictionaries making them easy to serialise or inspect in
tests.
"""

from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from typing import Any, Dict, Mapping


class ContextHub:
    """Central state store shared across subsystems."""

    def __init__(self) -> None:
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._subscribers: list[asyncio.Queue[Dict[str, Dict[str, Any]]]] = []

    # ------------------------------------------------------------------
    def update(
        self, component: str, payload: Mapping[str, Any], *, replace: bool = False
    ) -> None:
        """Merge ``payload`` into the state for ``component``.

        Parameters
        ----------
        component:
            Name of the subsystem publishing data (``"risk"``, ``"resources"`` …).
        payload:
            Mapping of simple values describing the current state.
        replace:
            When ``True`` the existing state for ``component`` is replaced
            entirely.  The default behaviour merges the payload into any
            existing dictionary which keeps incremental metrics lightweight.
        """

        data = dict(payload)
        with self._lock:
            if replace or component not in self._state:
                self._state[component] = data
            else:
                self._state[component].update(data)
            snapshot = self._snapshot_locked()
        self._broadcast(snapshot)

    # ------------------------------------------------------------------
    def remove(self, component: str) -> None:
        """Delete the stored state for ``component`` if present."""

        with self._lock:
            if component in self._state:
                self._state.pop(component, None)
            snapshot = self._snapshot_locked()
        self._broadcast(snapshot)

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a deep copy of the aggregated state."""

        with self._lock:
            return self._snapshot_locked()

    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue[Dict[str, Dict[str, Any]]]:
        """Return an asyncio queue receiving context snapshots."""

        queue: asyncio.Queue[Dict[str, Dict[str, Any]]] = asyncio.Queue()
        with self._lock:
            queue.put_nowait(self._snapshot_locked())
            self._subscribers.append(queue)
        return queue

    # ------------------------------------------------------------------
    def _snapshot_locked(self) -> Dict[str, Dict[str, Any]]:
        return {name: deepcopy(data) for name, data in self._state.items()}

    # ------------------------------------------------------------------
    def _broadcast(self, snapshot: Dict[str, Dict[str, Any]]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                continue
            except RuntimeError:
                # Queue was probably attached to a closed loop – drop it.
                with self._lock:
                    try:
                        self._subscribers.remove(queue)
                    except ValueError:
                        pass


# Global singleton used by the platform
context_hub = ContextHub()

__all__ = ["ContextHub", "context_hub"]

