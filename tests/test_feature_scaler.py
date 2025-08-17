import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from feature_scaler import FeatureScaler


def test_feature_scaler_streaming(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(loc=5.0, scale=2.0, size=(1000, 3)), columns=list("abc"))
    batches = np.array_split(df, 10)

    scaler = FeatureScaler()
    for batch in batches[:5]:
        scaler.partial_fit(batch)
    path = tmp_path / "scaler.pkl"
    scaler.save(path)

    scaler2 = FeatureScaler.load(path)
    for batch in batches[5:]:
        scaler2.partial_fit(batch)

    transformed = scaler2.transform(df)
    means = transformed.mean().values
    stds = transformed.std(ddof=0).values

    assert np.allclose(means, np.zeros(3), atol=1e-1)
    assert np.allclose(stds, np.ones(3), atol=1e-1)
