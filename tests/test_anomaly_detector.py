import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import Counter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.anomaly_detector import detect_anomalies


def test_detect_anomalies_zscore(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 2))
    df = pd.DataFrame(data, columns=["a", "b"])
    df.loc[0, "a"] = 15
    df.loc[1, "b"] = -15

    counter = Counter("anoms", "test anomalies")
    qfile = tmp_path / "quarantine.csv"
    clean, anomalies = detect_anomalies(
        df,
        method="zscore",
        threshold=3.0,
        quarantine_path=qfile,
        counter=counter,
    )

    assert len(anomalies) == 2
    assert len(clean) == len(df) - 2
    assert qfile.exists()
    assert counter._value.get() == 2
