import asyncio
import time
from pathlib import Path
import sys
import importlib.util

import numpy as np
import pandas as pd
import pytest

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
spec = importlib.util.spec_from_file_location(
    "data.live_recorder", root / "data" / "live_recorder.py"
)
live_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(live_module)
LiveRecorder = live_module.LiveRecorder
load_ticks = live_module.load_ticks
import types

sys.modules["requests"] = types.SimpleNamespace(get=lambda *a, **k: None)


def test_async_recording_order_and_latency(tmp_path: Path):
    async def _run():
        recorder = LiveRecorder(tmp_path, batch_size=5, flush_interval=0.01)
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        stop_event = asyncio.Event()

        async def produce():
            base = pd.Timestamp("2024-01-01", tz="UTC")
            for i in range(4):
                ts = base + pd.to_timedelta(np.arange(5) + i * 5, unit="s")
                df = pd.DataFrame(
                    {
                        "Timestamp": ts,
                        "Bid": np.arange(5, dtype=float) + i * 5,
                        "Ask": np.arange(5, dtype=float) + i * 5 + 0.1,
                        "BidVolume": np.ones(5),
                        "AskVolume": np.ones(5),
                    }
                )
                await in_q.put(df)
            await in_q.join()
            stop_event.set()

        batches: list[pd.DataFrame] = []

        async def consume():
            while True:
                batch = await out_q.get()
                if batch is None:
                    break
                batches.append(batch)

        start = time.perf_counter()
        await asyncio.gather(
            recorder.run(in_q, out_q, stop_event=stop_event),
            produce(),
            consume(),
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
        assert recorder.avg_latency() > 0

        recorded = load_ticks(tmp_path)
        combined = pd.concat(batches, ignore_index=True)
        pd.testing.assert_frame_equal(recorded, combined)

    asyncio.run(_run())
