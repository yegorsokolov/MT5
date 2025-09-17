import numpy as np
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import types

if "data.labels" not in sys.modules:
    def _label_fn(series, horizons):
        data = {f"direction_{h}": pd.Series(0, index=series.index, dtype=int) for h in horizons}
        return pd.DataFrame(data)

    labels_stub = types.SimpleNamespace(multi_horizon_labels=_label_fn)
    sys.modules["data.labels"] = labels_stub

if "data.streaming" not in sys.modules:
    def _stream_labels(chunks, horizons):
        for chunk in chunks:
            yield sys.modules["data.labels"].multi_horizon_labels(chunk["mid"], horizons)

    sys.modules["data.streaming"] = types.SimpleNamespace(stream_labels=_stream_labels)

from training.data_loader import load_training_frame
from training.labels import generate_training_labels
from training.postprocess import build_model_metadata, summarise_predictions
from training.utils import combined_sample_weight


class _DummyStrategy:
    def __init__(self) -> None:
        self.symbols: list[str] = []


def test_load_training_frame_override_returns_frame():
    df = pd.DataFrame({"a": [1, 2, 3]})
    cfg = SimpleNamespace(strategy=_DummyStrategy())
    result, source = load_training_frame(cfg, Path("."), df_override=df)
    pd.testing.assert_frame_equal(result, df)
    assert source == "override"


def test_generate_training_labels_stream_matches_offline():
    idx = pd.RangeIndex(10)
    mid = pd.Series(np.linspace(100, 101, len(idx)), index=idx)
    df = pd.DataFrame({"mid": mid})
    offline = generate_training_labels(df, stream=False, horizons=[1], chunk_size=5)
    streamed = generate_training_labels(df, stream=True, horizons=[1], chunk_size=4)
    pd.testing.assert_frame_equal(streamed, offline)


def test_combined_sample_weight_respects_quality_and_decay():
    y = np.array([0, 0, 1, 1], dtype=int)
    timestamps = np.array([0, 1, 2, 3], dtype=np.int64)
    dq = np.array([1.0, 0.5, 0.5, 1.0])
    weights = combined_sample_weight(y, timestamps, timestamps.max(), True, 2, dq)
    assert weights is not None
    assert weights[-1] > weights[1]
    assert weights[1] < weights[0]


def test_postprocess_helpers_round_trip():
    meta = build_model_metadata({0: 0.5}, interval_alpha=0.1, interval_coverage=0.9)
    assert meta["regime_thresholds"][0] == 0.5
    assert meta["interval_alpha"] == 0.1
    frame = summarise_predictions([0, 1], [0, 1], [0.3, 0.7], [0, 1], lower=[0.1, 0.2], upper=[0.9, 0.8])
    assert list(frame.columns) == ["y_true", "pred", "prob", "market_regime", "lower", "upper"]
