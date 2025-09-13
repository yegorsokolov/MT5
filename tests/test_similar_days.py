import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.similar_days import add_similar_day_features


def test_similarity_retrieval():
    df = pd.DataFrame({"return": np.arange(10) * 2.0})
    df, _ = add_similar_day_features(df, ["return"], "return", k=3)
    expected_mean = df["return"].iloc[2:5].mean()
    expected_std = df["return"].iloc[2:5].std(ddof=0)
    assert df.loc[5, "nn_return_mean"] == expected_mean
    assert df.loc[5, "nn_vol"] == expected_std


def test_nn_features_improve_metric():
    rng = np.random.default_rng(0)
    n = 200
    regime = np.where((np.arange(n) // 20) % 2 == 0, 1.0, -1.0)
    returns = regime + rng.normal(0, 0.1, size=n)
    noise = rng.normal(size=n)
    labels = (regime > 0).astype(int)
    df = pd.DataFrame({"return": returns, "noise": noise, "label": labels})
    df, _ = add_similar_day_features(df, ["return"], "return", k=5)

    split = int(n * 0.7)
    test_df = df.iloc[split:]
    acc_base = (np.sign(test_df["noise"]) > 0).astype(int).eq(test_df["label"]).mean()
    acc_nn = (test_df["nn_return_mean"] > 0).astype(int).eq(test_df["label"]).mean()

    assert acc_nn > acc_base
