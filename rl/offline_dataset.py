from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from event_store import EventStore


@dataclass
class Experience:
    """Single offline reinforcement learning transition."""

    obs: Sequence[float]
    action: Sequence[float]
    reward: float
    next_obs: Sequence[float]
    done: bool


class OfflineDataset:
    """Load experiences from the :mod:`event_store`.

    Events are expected to have type ``"experience"`` with payload
    containing ``obs``, ``action``, ``reward``, ``next_obs`` and ``done``
    fields.  The dataset keeps the transitions in memory and can yield
    mini-batches for simple offline training loops.
    """

    def __init__(self, store: EventStore | str | Path | None = None) -> None:
        if isinstance(store, (str, Path)):
            self.store = EventStore(store)
            self._close = True
        elif store is None:
            self.store = EventStore()
            self._close = True
        else:
            self.store = store
            self._close = False

        self.samples: List[Experience] = []
        for ev in self.store.iter_events("experience"):
            pl = ev["payload"]
            self.samples.append(
                Experience(
                    obs=pl.get("obs", []),
                    action=pl.get("action", []),
                    reward=float(pl.get("reward", 0.0)),
                    next_obs=pl.get("next_obs", []),
                    done=bool(pl.get("done", False)),
                )
            )

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.samples)

    def iter_batches(self, batch_size: int) -> Iterable[Tuple[List, List, List, List, List]]:
        """Yield mini-batches of experiences."""
        for i in range(0, len(self.samples), batch_size):
            batch = self.samples[i : i + batch_size]
            yield (
                [s.obs for s in batch],
                [s.action for s in batch],
                [s.reward for s in batch],
                [s.next_obs for s in batch],
                [s.done for s in batch],
            )

    def close(self) -> None:  # pragma: no cover - trivial
        if self._close:
            self.store.close()


__all__ = ["OfflineDataset", "Experience"]
