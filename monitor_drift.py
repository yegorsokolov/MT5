from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import ks_2samp

import metrics
try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests
    def send_alert(msg: str) -> None:  # type: ignore
        return

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

DRIFT_METRICS = LOG_DIR / "drift_metrics.parquet"
BASELINE_METRICS = LOG_DIR / "training_baseline.parquet"


class DriftMonitor:
    """Capture feature/prediction distributions and detect drift."""

    def __init__(
        self,
        baseline_path: Path = BASELINE_METRICS,
        store_path: Path = DRIFT_METRICS,
        threshold: float = 0.05,
    ) -> None:
        self.baseline_path = baseline_path
        self.store_path = store_path
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        self._task: Optional[asyncio.Task] = None

    def record(self, features: pd.DataFrame, preds: pd.Series) -> None:
        """Append feature and prediction samples to the drift store."""
        df = features.copy()
        df["prediction"] = preds
        if self.store_path.exists():
            df.to_parquet(self.store_path, engine="pyarrow", append=True)
        else:
            df.to_parquet(self.store_path, engine="pyarrow")

    def compare(self) -> None:
        """Compare current distributions against training baselines."""
        if not (self.baseline_path.exists() and self.store_path.exists()):
            return
        baseline = pd.read_parquet(self.baseline_path)
        current = pd.read_parquet(self.store_path)
        for col in baseline.columns:
            if col not in current.columns:
                continue
            stat, pval = ks_2samp(baseline[col].dropna(), current[col].dropna())
            if pval < self.threshold:
                self.logger.warning(
                    "Data drift detected for %s (p=%.4f)", col, pval
                )
                metrics.DRIFT_EVENTS.inc()
                send_alert(
                    f"Data drift detected for {col} (p={pval:.4f})"
                )

    async def _periodic_check(self) -> None:
        while True:
            await asyncio.sleep(24 * 60 * 60)
            self.compare()

    def start(self) -> None:
        """Start daily drift monitoring."""
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._periodic_check())


monitor = DriftMonitor()
monitor.start()
