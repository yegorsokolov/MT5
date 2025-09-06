import asyncio
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import message_bus
from core.distributed_scheduler import DistributedScheduler


async def _next_message(bus, topic):
    async for msg in bus.subscribe(topic):
        return msg


@pytest.mark.asyncio
async def test_job_migrates_to_strongest_vm():
    # fresh message bus
    message_bus._message_bus = None
    bus = message_bus.get_message_bus()
    scheduler = DistributedScheduler(bus)
    scheduler.start()

    # broadcast node capabilities
    await bus.publish(
        scheduler.CAP_TOPIC,
        {
            "node": "weak",
            "capabilities": {"cpus": 2, "memory_gb": 4, "has_gpu": False, "bandwidth": 50},
        },
    )
    await bus.publish(
        scheduler.CAP_TOPIC,
        {
            "node": "strong",
            "capabilities": {
                "cpus": 8,
                "memory_gb": 32,
                "has_gpu": True,
                "gpu_count": 1,
                "bandwidth": 100,
            },
        },
    )
    # allow scheduler to process broadcasts
    await asyncio.sleep(0)

    job_future = asyncio.create_task(_next_message(bus, f"{scheduler.JOB_PREFIX}strong"))
    node = await scheduler.dispatch({"type": "backtest", "requirements": {"cpus": 2}})
    assert node == "strong"
    msg = await asyncio.wait_for(job_future, 1)
    assert msg["type"] == "backtest"


@pytest.mark.asyncio
async def test_fallback_to_local_execution():
    message_bus._message_bus = None
    bus = message_bus.get_message_bus()
    scheduler = DistributedScheduler(bus)
    scheduler.start()

    await bus.publish(
        scheduler.CAP_TOPIC,
        {"node": "weak", "capabilities": {"cpus": 2, "memory_gb": 4, "has_gpu": False}},
    )
    await asyncio.sleep(0)

    result = await scheduler.dispatch(
        {
            "type": "search",
            "requirements": {"has_gpu": True},
            "local_fallback": lambda: "local",
        }
    )
    assert result == "local"
