from __future__ import annotations

"""Simple learning component for execution parameter tuning."""

import json
import threading
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

from .fill_history import load_history

PARAMS_FILE = Path("execution_params.json")


class OptimizationLoopHandle:
    """Handle for the background optimisation loop."""

    def __init__(self, thread: threading.Thread, stop_event: threading.Event) -> None:
        self._thread = thread
        self._stop_event = stop_event

    def stop(self) -> None:
        """Signal the optimisation loop to stop."""

        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the optimisation thread to exit."""

        self._thread.join(timeout)

    def is_alive(self) -> bool:
        """Return whether the optimisation thread is still running."""

        return self._thread.is_alive()


class ExecutionOptimizer:
    """Optimize order placement parameters from historical fills."""

    def __init__(self, history_file: Path = Path("fill_history.csv"), params_file: Path = PARAMS_FILE) -> None:
        self.history_file = Path(history_file)
        self.params_file = Path(params_file)

    # ------------------------------------------------------------------
    def optimize(self) -> Dict[str, float | None]:
        """Re-compute optimal parameters from fill history."""
        history = load_history(self.history_file)
        if not history:
            params = {"limit_offset": 0.0, "slice_size": None}
        else:
            avg_slippage = mean(rec.slippage for rec in history)
            avg_depth = mean(rec.depth for rec in history)
            limit_offset = avg_slippage
            slice_size = max(1.0, avg_depth / 2.0)
            params = {"limit_offset": limit_offset, "slice_size": slice_size}
        with self.params_file.open("w") as f:
            json.dump(params, f)
        return params

    # ------------------------------------------------------------------
    def get_params(self) -> Dict[str, float | None]:
        """Return the last optimized parameters."""
        if self.params_file.exists():
            with self.params_file.open() as f:
                return json.load(f)
        return {"limit_offset": 0.0, "slice_size": None}

    # ------------------------------------------------------------------
    def schedule_nightly(self) -> OptimizationLoopHandle:
        """Run optimization once every 24h in a background thread."""

        stop_event = threading.Event()

        def loop() -> None:
            while not stop_event.is_set():
                try:
                    self.optimize()
                except Exception:
                    pass
                if stop_event.wait(24 * 60 * 60):
                    break

        thread = threading.Thread(
            target=loop,
            name="ExecutionOptimizerNightly",
            daemon=True,
        )
        thread.start()
        return OptimizationLoopHandle(thread, stop_event)
