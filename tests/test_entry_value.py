import asyncio
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
import types

# Ensure environment module used by signal_queue is present
sys.modules["utils.environment"] = types.SimpleNamespace(ensure_environment=lambda: None)
sys.modules.setdefault("mlflow", types.SimpleNamespace())

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mt5 import signal_queue
import risk.position_sizer as ps
from analysis.entry_value import EntryValueScorer, log_entry_value


def test_scorer_and_logging(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTRY_VALUE_REPORT_PATH", str(tmp_path))
    scorer = EntryValueScorer(cost=0.0)
    score = scorer.score(0.5, 0.1, [0.2, -0.1])
    assert score > 0
    log_entry_value("2024", "AAA", score, 0.05)
    log_file = tmp_path / "entry_value.csv"
    assert log_file.exists()
    df = pd.read_csv(log_file)
    assert pytest.approx(df.iloc[0]["predicted"]) == score
    assert pytest.approx(df.iloc[0]["realised"]) == 0.05


@pytest.mark.asyncio
async def test_signal_queue_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTRY_VALUE_REPORT_PATH", str(tmp_path))
    monkeypatch.setattr(signal_queue, "record_metric", lambda *a, **k: None)
    monkeypatch.setattr(ps, "record_metric", lambda *a, **k: None)
    monkeypatch.setattr(signal_queue._EVENT_STORE, "record", lambda *a, **k: None)

    scorer = EntryValueScorer()
    async with signal_queue.get_async_publisher("tcp://127.0.0.1:6012") as pub, signal_queue.get_async_subscriber(
        "tcp://127.0.0.1:6012"
    ) as sub:
        df = pd.DataFrame(
            {
                "Timestamp": ["T1", "T2"],
                "prob": [0.6, 0.6],
                "confidence": [1.0, 1.0],
                "depth_imbalance": [-0.5, 0.5],
                "volatility_30": [0.1, 0.1],
                "regime_embed": [[0.0], [0.0]],
                "future_return": [-0.02, 0.03],
            }
        )
        await signal_queue.publish_dataframe_async(pub, df, fmt="json")
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub, fmt="json", entry_scorer=scorer)
        out = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert out["expected_value"] > 0
        assert out["prob"] == 0.6
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), timeout=0.2)

    log_file = tmp_path / "entry_value.csv"
    df_log = pd.read_csv(log_file)
    assert len(df_log) == 2
    assert df_log["predicted"].iloc[0] < 0
    assert df_log["predicted"].iloc[1] > 0
