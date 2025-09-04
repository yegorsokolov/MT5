import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from analysis.tick_anomaly_detector import TickAnomalyDetector


def _make_stream():
    start = pd.Timestamp("2023-01-01")
    rows = []
    for i in range(20):
        ts = start + pd.Timedelta(seconds=i)
        bid = 1.0 + i * 0.0001
        ask = bid + 0.0002
        rows.append((ts, bid, ask, 1, 1))
    # spread outlier
    rows.append((start + pd.Timedelta(seconds=20), 1.002, 1.102, 1, 1))
    # timestamp glitch (non-monotonic)
    rows.append((start + pd.Timedelta(seconds=10), 1.003, 1.0032, 1, 1))
    # price jump outlier
    rows.append((start + pd.Timedelta(seconds=21), 2.0, 2.0002, 1, 1))
    df = pd.DataFrame(rows, columns=["Timestamp", "Bid", "Ask", "BidVolume", "AskVolume"])
    return df


def test_detector_filters_anomalies():
    df = _make_stream()
    det = TickAnomalyDetector()
    clean, anomalies = det.filter(df)
    assert anomalies == 3
    assert len(clean) == len(df) - 3
    assert clean["Timestamp"].is_monotonic_increasing
