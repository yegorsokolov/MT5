from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

from services.message_bus import MessageBus, get_message_bus


@dataclass
class NodeCapabilities:
    """Basic resource information advertised by a compute node."""

    cpus: int
    memory_gb: int
    has_gpu: bool
    bandwidth: float
    gpu_count: int = 0

    def score(self) -> tuple:
        """Return a tuple used to rank nodes by strength."""

        return (
            int(self.has_gpu),
            self.gpu_count,
            self.bandwidth,
            self.cpus,
            self.memory_gb,
        )


class DistributedScheduler:
    """Dispatch jobs to the most capable available node."""

    CAP_TOPIC = "node_capabilities"
    JOB_PREFIX = "jobs."

    def __init__(self, bus: MessageBus | None = None) -> None:
        self.bus = bus or get_message_bus()
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, NodeCapabilities] = {}
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Begin listening for capability broadcasts."""

        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - event loop not running
            loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._listen_capabilities())

    # ------------------------------------------------------------------
    async def _listen_capabilities(self) -> None:
        """Maintain an updated view of cluster node capabilities."""

        async for msg in self.bus.subscribe(self.CAP_TOPIC):
            try:
                node = msg.get("node")
                caps = msg.get("capabilities", {})
                if not node:
                    continue
                self.nodes[node] = NodeCapabilities(
                    cpus=int(caps.get("cpus", 0)),
                    memory_gb=int(caps.get("memory_gb", 0)),
                    has_gpu=bool(caps.get("has_gpu", False)),
                    bandwidth=float(caps.get("bandwidth", 0)),
                    gpu_count=int(caps.get("gpu_count", caps.get("gpus", 0))),
                )
            except Exception:  # pragma: no cover - defensive
                self.logger.exception("Invalid capability broadcast: %s", msg)

    # ------------------------------------------------------------------
    def _meets(self, caps: NodeCapabilities, req: Dict[str, Any]) -> bool:
        if caps.cpus < req.get("cpus", 0):
            return False
        if caps.memory_gb < req.get("memory_gb", 0):
            return False
        if caps.bandwidth < req.get("bandwidth", 0):
            return False
        if req.get("has_gpu") and not caps.has_gpu:
            return False
        if caps.gpu_count < req.get("gpu_count", 0):
            return False
        return True

    # ------------------------------------------------------------------
    def _select_node(self, req: Dict[str, Any]) -> Optional[str]:
        best_node: Optional[str] = None
        best_score: tuple | None = None
        for node, caps in self.nodes.items():
            if not self._meets(caps, req):
                continue
            score = caps.score()
            if best_score is None or score > best_score:
                best_score = score
                best_node = node
        return best_node

    # ------------------------------------------------------------------
    async def dispatch(self, job: Dict[str, Any]) -> Any:
        """Dispatch ``job`` to the strongest suitable node.

        If no remote nodes meet ``job['requirements']`` and ``local_fallback`` is
        provided, the callback is executed locally instead.
        """

        req = job.get("requirements", {})
        node = self._select_node(req)
        if node:
            await self.bus.publish(f"{self.JOB_PREFIX}{node}", job)
            return node
        fallback: Optional[Callable[[], Any]] = job.get("local_fallback")
        if fallback:
            result = fallback()
            if asyncio.iscoroutine(result):
                return await result
            return result
        raise RuntimeError("No suitable node for job and no local fallback provided")


__all__ = ["DistributedScheduler", "NodeCapabilities"]
