import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import psutil
try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@dataclass
class ResourceCapabilities:
    """Basic view of available system resources."""

    cpus: int
    memory_gb: float
    has_gpu: bool
    gpu_count: int

    def model_size(self) -> str:
        """Return "full" for capable machines else "lite"."""
        if self.cpus >= 4 and self.memory_gb >= 16 and self.has_gpu:
            return "full"
        return "lite"

    def ddp(self) -> bool:
        """Return True if DistributedDataParallel should be enabled."""
        return self.gpu_count > 1


class ResourceMonitor:
    """Monitor and periodically refresh system resource info."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.capabilities = self._probe()
        self._task: Optional[asyncio.Task] = None

    def _probe(self) -> ResourceCapabilities:
        cpus = psutil.cpu_count(logical=True) or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        gpu_count = int(torch.cuda.device_count()) if torch and torch.cuda else 0
        has_gpu = gpu_count > 0
        return ResourceCapabilities(
            cpus=cpus, memory_gb=memory_gb, has_gpu=has_gpu, gpu_count=gpu_count
        )

    async def _periodic_probe(self) -> None:
        while True:
            await asyncio.sleep(24 * 60 * 60)
            self.capabilities = self._probe()
            self.logger.info("Refreshed resource capabilities: %s", self.capabilities)

    def start(self) -> None:
        """Start periodic background probing."""
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._periodic_probe())


monitor = ResourceMonitor()
