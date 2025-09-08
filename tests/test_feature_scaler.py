import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from feature_scaler import FeatureScaler


def test_feature_scaler_streaming(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(loc=5.0, scale=2.0, size=(1000, 3)), columns=list("abc")
    )
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


def test_feature_scaler_clipping():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1000, 1))
    scaler = FeatureScaler(clip_pct=1.0)
    scaler.partial_fit(data)

    clip_min = scaler.clip_min_[0]
    clip_max = scaler.clip_max_[0]
    assert np.isclose(clip_min, np.percentile(data, 1))
    assert np.isclose(clip_max, np.percentile(data, 99))

    extreme = np.array([[clip_max + 10], [clip_min - 10]])
    reference = np.array([[clip_max], [clip_min]])
    transformed_extreme = scaler.transform(extreme)
    transformed_reference = scaler.transform(reference)

    assert np.allclose(transformed_extreme, transformed_reference)


def test_feature_scaler_median_iqr_scaling():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list("abc"))
    df.loc[0, "a"] = 100
    df.loc[1, "b"] = -100

    scaler = FeatureScaler(use_median=True)
    scaler.partial_fit(df)
    transformed = scaler.transform(df)

    medians = transformed.median().values
    q1 = transformed.quantile(0.25).values
    q3 = transformed.quantile(0.75).values
    iqr = q3 - q1

    assert np.allclose(medians, np.zeros(3), atol=1e-6)
    assert np.allclose(iqr, np.ones(3), atol=1e-6)
