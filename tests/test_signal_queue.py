import asyncio
from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import signal_queue
import risk_manager


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


def test_credible_interval_adjust_size():
    queue = signal_queue.get_signal_backend({})
    df = pd.DataFrame(
        {
            "Timestamp": ["2022"],
            "pred": [0.05],
            "ci_lower": [0.01],
            "ci_upper": [0.09],
        }
    )
    signal_queue.publish_dataframe(queue, df)
    gen = signal_queue.iter_messages(queue)
    msg = asyncio.run(gen.__anext__())
    assert "prediction" in msg
    ci = (msg["prediction"]["lower"], msg["prediction"]["upper"])
    rm = risk_manager.RiskManager(max_drawdown=1.0)
    sized = rm.adjust_size("EURUSD", 1.0, pd.Timestamp("2022"), 1, cred_interval=ci)
    assert sized < 1.0
