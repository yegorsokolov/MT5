import asyncio
import sys
import time
from typing import Callable

import pytest

try:  # pragma: no cover - uvloop is optional
    import uvloop
except Exception:  # pragma: no cover - uvloop may be missing
    uvloop = None


@pytest.mark.skipif(sys.platform.startswith("win"), reason="uvloop unsupported on Windows")
def test_uvloop_benchmark() -> None:
    if uvloop is None:
        pytest.skip("uvloop not installed")

    async def runner() -> None:
        await asyncio.sleep(0)

    N = 50_000

    def run(factory: Callable[[], asyncio.AbstractEventLoop]) -> float:
        loop = factory()
        asyncio.set_event_loop(loop)
        start = time.perf_counter()
        for _ in range(N):
            loop.call_soon(lambda: None)
        loop.run_until_complete(runner())
        duration = time.perf_counter() - start
        loop.close()
        asyncio.set_event_loop(None)
        return duration

    default_time = run(asyncio.new_event_loop)
    uvloop_time = run(uvloop.new_event_loop)
    assert uvloop_time < default_time

