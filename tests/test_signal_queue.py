import asyncio
import pandas as pd
import zmq
import pytest
from pathlib import Path
import sys
import types

sys.modules["utils.environment"] = types.SimpleNamespace(ensure_environment=lambda: None)

sys.path.append(str(Path(__file__).resolve().parents[1]))

import signal_queue
import risk.position_sizer as ps
from risk.position_sizer import PositionSizer
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ensemble", Path(__file__).resolve().parents[1] / "models" / "ensemble.py"
)
ensemble_mod = importlib.util.module_from_spec(spec)
sys.modules["ensemble"] = ensemble_mod
spec.loader.exec_module(ensemble_mod)
EnsembleModel = ensemble_mod.EnsembleModel


def test_publish_and_receive(monkeypatch):
    calls = []
    def fake_record(name, val, tags=None):
        calls.append((name, val, tags))
    monkeypatch.setattr(signal_queue, "record_metric", fake_record)
    monkeypatch.setattr(ps, "record_metric", fake_record)
    with signal_queue.get_publisher("tcp://127.0.0.1:6000") as pub, signal_queue.get_subscriber(
        "tcp://127.0.0.1:6000"
    ) as sub:
        df = pd.DataFrame({"Timestamp": ["2020"], "prob": [0.8], "confidence": [0.9]})
        signal_queue.publish_dataframe(pub, df)
        raw = sub.recv()
        msg = signal_queue.signals_pb2.Signal()
        msg.ParseFromString(raw)
        assert float(msg.probability) == 0.8
        assert pytest.approx(msg.confidence) == 0.9
        assert any(c[0] == "queue_depth" for c in calls)
    assert pub.closed
    assert sub.closed


def test_publish_and_receive_json(monkeypatch):
    def fake_record(name, val, tags=None):
        pass
    monkeypatch.setattr(signal_queue, "record_metric", fake_record)
    monkeypatch.setattr(ps, "record_metric", fake_record)
    with signal_queue.get_publisher("tcp://127.0.0.1:6002") as pub, signal_queue.get_subscriber(
        "tcp://127.0.0.1:6002"
    ) as sub:
        df = pd.DataFrame({"Timestamp": ["2021"], "prob": [0.7], "confidence": [0.8]})
        signal_queue.publish_dataframe(pub, df, fmt="json")
        data = sub.recv_json()
        assert data["prob"] == 0.7
        assert data["confidence"] == 0.8
    assert pub.closed
    assert sub.closed


@pytest.mark.asyncio
async def test_async_publish_and_iter(monkeypatch):
    calls = []
    def fake_record(name, val, tags=None):
        calls.append((name, val, tags))
    monkeypatch.setattr(signal_queue, "record_metric", fake_record)
    monkeypatch.setattr(ps, "record_metric", fake_record)
    async with signal_queue.get_async_publisher(
        "tcp://127.0.0.1:6001"
    ) as pub, signal_queue.get_async_subscriber("tcp://127.0.0.1:6001") as sub:
        df = pd.DataFrame({"Timestamp": ["2020"], "prob": [0.9], "confidence": [0.6]})
        await signal_queue.publish_dataframe_async(pub, df)
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub)
        out = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert out["prob"] == 0.9
        assert pytest.approx(out["confidence"]) == 0.6
        # consumer combines model outputs via EnsembleModel
        class _Const:
            def __init__(self, p: float) -> None:
                self.p = p

            def predict_proba(self, X):  # pragma: no cover - trivial
                import numpy as np

                return np.column_stack([1 - self.p, np.full(len(X), self.p)])

        ens = EnsembleModel({"a": _Const(out["prob"]), "b": _Const(0.5)})
        preds = ens.predict(pd.DataFrame({"x": [0]}))
        assert pytest.approx(preds["ensemble"][0]) == (out["prob"] + 0.5) / 2
        assert any(c[0] == "queue_depth" for c in calls)
    assert pub.closed
    assert sub.closed


@pytest.mark.asyncio
async def test_async_publish_and_iter_json(monkeypatch):
    def fake_record(name, val, tags=None):
        pass
    monkeypatch.setattr(signal_queue, "record_metric", fake_record)
    monkeypatch.setattr(ps, "record_metric", fake_record)
    async with signal_queue.get_async_publisher(
        "tcp://127.0.0.1:6003"
    ) as pub, signal_queue.get_async_subscriber("tcp://127.0.0.1:6003") as sub:
        df = pd.DataFrame({"Timestamp": ["2022"], "prob": [0.6], "confidence": [0.7]})
        await signal_queue.publish_dataframe_async(pub, df, fmt="json")
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub, fmt="json")
        out = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert out["prob"] == 0.6
        assert out["confidence"] == 0.7
    assert pub.closed
    assert sub.closed


@pytest.mark.asyncio
async def test_iter_with_position_sizer(monkeypatch):
    def fake_record(name, val, tags=None):
        pass
    monkeypatch.setattr(signal_queue, "record_metric", fake_record)
    monkeypatch.setattr(ps, "record_metric", fake_record)
    sizer = PositionSizer(capital=1000.0)
    async with signal_queue.get_async_publisher(
        "tcp://127.0.0.1:6004"
    ) as pub, signal_queue.get_async_subscriber("tcp://127.0.0.1:6004") as sub:
        df = pd.DataFrame({"Timestamp": ["2023"], "prob": [0.8], "confidence": [0.5]})
        await signal_queue.publish_dataframe_async(pub, df)
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub, sizer=sizer)
        out = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert out["confidence"] == 0.5
        assert pytest.approx(out["size"]) == 300.0

