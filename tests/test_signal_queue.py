import asyncio
import pandas as pd
import zmq
import pytest
from prometheus_client import Gauge

import signal_queue


def test_publish_and_receive(monkeypatch):
    gauge = Gauge("qd_test", "queue")
    monkeypatch.setattr(signal_queue, "QUEUE_DEPTH", gauge)
    with signal_queue.get_publisher("tcp://127.0.0.1:6000") as pub, signal_queue.get_subscriber(
        "tcp://127.0.0.1:6000"
    ) as sub:
        df = pd.DataFrame({"Timestamp": ["2020"], "prob": [0.8]})
        signal_queue.publish_dataframe(pub, df)
        raw = sub.recv()
        msg = signal_queue.signals_pb2.Signal()
        msg.ParseFromString(raw)
        assert float(msg.probability) == 0.8
        assert gauge._value.get() == 1
    assert pub.closed
    assert sub.closed


def test_publish_and_receive_json(monkeypatch):
    gauge = Gauge("qd_json", "queue")
    monkeypatch.setattr(signal_queue, "QUEUE_DEPTH", gauge)
    with signal_queue.get_publisher("tcp://127.0.0.1:6002") as pub, signal_queue.get_subscriber(
        "tcp://127.0.0.1:6002"
    ) as sub:
        df = pd.DataFrame({"Timestamp": ["2021"], "prob": [0.7]})
        signal_queue.publish_dataframe(pub, df, fmt="json")
        data = sub.recv_json()
        assert data["prob"] == 0.7
    assert pub.closed
    assert sub.closed


@pytest.mark.asyncio
async def test_async_publish_and_iter(monkeypatch):
    gauge = Gauge("qd_test2", "queue")
    monkeypatch.setattr(signal_queue, "QUEUE_DEPTH", gauge)
    async with signal_queue.get_async_publisher(
        "tcp://127.0.0.1:6001"
    ) as pub, signal_queue.get_async_subscriber("tcp://127.0.0.1:6001") as sub:
        df = pd.DataFrame({"Timestamp": ["2020"], "prob": [0.9]})
        await signal_queue.publish_dataframe_async(pub, df)
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub)
        out = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert out["prob"] == 0.9
        assert gauge._value.get() == 0
    assert pub.closed
    assert sub.closed


@pytest.mark.asyncio
async def test_async_publish_and_iter_json(monkeypatch):
    gauge = Gauge("qd_json_async", "queue")
    monkeypatch.setattr(signal_queue, "QUEUE_DEPTH", gauge)
    async with signal_queue.get_async_publisher(
        "tcp://127.0.0.1:6003"
    ) as pub, signal_queue.get_async_subscriber("tcp://127.0.0.1:6003") as sub:
        df = pd.DataFrame({"Timestamp": ["2022"], "prob": [0.6]})
        await signal_queue.publish_dataframe_async(pub, df, fmt="json")
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub, fmt="json")
        out = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert out["prob"] == 0.6
    assert pub.closed
    assert sub.closed

