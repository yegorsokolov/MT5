import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.inference_latency import InferenceLatency


def test_moving_average() -> None:
    tracker = InferenceLatency(window=3)
    tracker.record("m", 1.0)
    tracker.record("m", 2.0)
    assert tracker.moving_average("m") == 1.5
    tracker.record("m", 3.0)
    assert tracker.moving_average("m") == 2.0
    tracker.record("m", 4.0)
    assert tracker.moving_average("m") == 3.0
