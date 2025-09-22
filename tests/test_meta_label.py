import asyncio
import numpy as np
import pandas as pd
import pytest
from prometheus_client import Gauge
from pathlib import Path
import importlib.util
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mt5 import signal_queue

spec = importlib.util.spec_from_file_location(
    "meta_label", Path(__file__).resolve().parents[1] / "models" / "meta_label.py"
)
meta_label_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta_label_mod)
train_meta_classifier = meta_label_mod.train_meta_classifier


def test_meta_classifier_reduces_false_positives():
    probs = np.linspace(0, 1, 200)
    y_true = (probs > 0.7).astype(int)
    features = pd.DataFrame({"prob": probs, "confidence": np.ones_like(probs)})
    clf = train_meta_classifier(features, y_true)

    preds = probs > 0.5
    primary_fp = ((preds == 1) & (y_true == 0)).sum()
    meta_preds = clf.predict(features[preds])
    filtered_fp = ((meta_preds == 1) & (y_true[preds] == 0)).sum()
    assert filtered_fp < primary_fp


@pytest.mark.asyncio
async def test_signal_queue_meta_filter(monkeypatch):
    probs = np.linspace(0, 1, 200)
    y_true = (probs > 0.7).astype(int)
    features = pd.DataFrame({"prob": probs, "confidence": np.ones_like(probs)})
    clf = train_meta_classifier(features, y_true)

    gauge = Gauge("qd_meta", "queue")
    monkeypatch.setattr(signal_queue, "QUEUE_DEPTH", gauge)
    async with signal_queue.get_async_publisher("tcp://127.0.0.1:6020") as pub, \
        signal_queue.get_async_subscriber("tcp://127.0.0.1:6020") as sub:
        df_send = pd.DataFrame({
            "Timestamp": ["t1", "t2"],
            "prob": [0.55, 0.9],
            "confidence": [1.0, 1.0],
        })
        await signal_queue.publish_dataframe_async(pub, df_send)
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub, meta_clf=clf)
        out = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert out["prob"] == 0.9
