import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List

from utils.resource_monitor import monitor
from model_registry import select_models
from features import get_feature_pipeline


class EnergySaver:
    """Policy layer that reacts to high resource usage by throttling.

    The saver monitors CPU and power proxy metrics from
    :mod:`utils.resource_monitor` and when thresholds are exceeded it attempts to
    lighten the workload by refreshing the model registry (which may swap heavy
    models for remote/quantised variants) and re-evaluating the feature pipeline
    to favour lighter feature sets.  Decisions are persisted under
    ``reports/energy/latest.json`` for dashboard consumption.
    """

    def __init__(
        self,
        cpu_threshold: float = 90.0,
        power_threshold: float = 90.0,
    ) -> None:
        self.cpu_threshold = cpu_threshold
        self.power_threshold = power_threshold
        self.logger = logging.getLogger(__name__)
        self._task: asyncio.Task | None = None
        self._report_dir = Path("reports/energy")

    async def _run(self) -> None:
        q = monitor.subscribe_usage()
        while True:
            usage = await q.get()
            cpu = float(usage.get("cpu_pct", 0.0))
            power = float(usage.get("power_proxy", cpu))
            decisions: List[str] = []
            if cpu > self.cpu_threshold or power > self.power_threshold:
                try:
                    select_models()
                    decisions.append("remote_inference")
                except Exception:
                    self.logger.debug("Model refresh failed", exc_info=True)
                try:
                    get_feature_pipeline()
                    decisions.append("light_features")
                except Exception:
                    self.logger.debug("Feature refresh failed", exc_info=True)
            self._write_report(usage, decisions)

    def start(self) -> None:
        """Start monitoring for high resource usage."""

        if self._task is not None:
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def _write_report(self, usage: Dict[str, object], decisions: List[str]) -> None:
        try:
            self._report_dir.mkdir(parents=True, exist_ok=True)
            data = {"usage": usage, "decisions": decisions}
            with (self._report_dir / "latest.json").open("w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            self.logger.debug("Failed to write energy report", exc_info=True)


saver = EnergySaver()
