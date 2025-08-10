import importlib.util
import sys
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "metrics", Path(__file__).resolve().parents[1] / "metrics.py"
)
metrics = importlib.util.module_from_spec(spec)
sys.modules["metrics"] = metrics
spec.loader.exec_module(metrics)

spec = importlib.util.spec_from_file_location(
    "data.sanitize", Path(__file__).resolve().parents[1] / "data" / "sanitize.py"
)
sanitize_mod = importlib.util.module_from_spec(spec)
sys.modules["data.sanitize"] = sanitize_mod
spec.loader.exec_module(sanitize_mod)

TICK_ANOMALIES = metrics.TICK_ANOMALIES
sanitize_ticks = sanitize_mod.sanitize_ticks


def reset_metrics():
    for label in ["non_monotonic", "spread", "price_jump"]:
        TICK_ANOMALIES.labels(label)._value.set(0)


def test_remove_non_monotonic_and_spread():
    reset_metrics()
    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:01",
                "2024-01-01 00:00:01",
                "2024-01-01 00:00:02",
            ]),
            "Bid": [1.0, 1.0, 1.0, 1.0],
            "Ask": [1.0002, 1.08, 0.9, 1.0002],
            "BidVolume": [1, 1, 1, 1],
            "AskVolume": [1, 1, 1, 1],
        }
    )
    out = sanitize_ticks(df, price_jump_threshold=0.1, max_spread=0.05)
    assert len(out) == 2
    assert TICK_ANOMALIES.labels("non_monotonic")._value.get() == 1
    assert TICK_ANOMALIES.labels("spread")._value.get() == 2
    assert TICK_ANOMALIES.labels("price_jump")._value.get() == 0


def test_price_jump_filter():
    reset_metrics()
    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:01",
                "2024-01-01 00:00:02",
            ]),
            "Bid": [1.0, 1.0, 1.5],
            "Ask": [1.0002, 1.0002, 1.5002],
            "BidVolume": [1, 1, 1],
            "AskVolume": [1, 1, 1],
        }
    )
    out = sanitize_ticks(df, price_jump_threshold=0.1, max_spread=0.05)
    assert len(out) == 2
    assert TICK_ANOMALIES.labels("non_monotonic")._value.get() == 0
    assert TICK_ANOMALIES.labels("spread")._value.get() == 0
    assert TICK_ANOMALIES.labels("price_jump")._value.get() == 1
