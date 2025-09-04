from __future__ import annotations

"""Simple learning component for execution parameter tuning."""

import json
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

from .fill_history import load_history

PARAMS_FILE = Path("execution_params.json")


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
    def schedule_nightly(self) -> None:
        """Run optimization once every 24h in a background thread."""
        import threading
        import time

        def loop() -> None:
            while True:
                try:
                    self.optimize()
                except Exception:
                    pass
                time.sleep(24 * 60 * 60)

        threading.Thread(target=loop, daemon=True).start()
