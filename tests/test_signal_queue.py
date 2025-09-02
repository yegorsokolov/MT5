import asyncio
from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import signal_queue


def test_publish_and_iter():
    queue = signal_queue.get_signal_backend({})
    df = pd.DataFrame({"Timestamp": ["2020"], "prob": [0.8], "confidence": [0.9]})
    signal_queue.publish_dataframe(queue, df)
    gen = signal_queue.iter_messages(queue)
    msg = asyncio.run(gen.__anext__())
    assert msg["prob"] == 0.8
    assert pytest.approx(msg["confidence"]) == 0.9


def test_async_publish_and_iter():
    queue = signal_queue.get_signal_backend({})
    df = pd.DataFrame({"Timestamp": ["2021"], "prob": [0.7], "confidence": [0.8]})
    asyncio.run(signal_queue.publish_dataframe_async(queue, df))
    gen = signal_queue.iter_messages(queue)
    msg = asyncio.run(asyncio.wait_for(gen.__anext__(), timeout=1))
    assert msg["prob"] == 0.7
    assert pytest.approx(msg["confidence"]) == 0.8
