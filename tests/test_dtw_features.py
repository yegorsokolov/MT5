import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.dtw_features import compute, add_dtw_features


def test_dtw_features_improve_validation():
    import sys

    sys.modules.pop("scipy", None)
    sys.modules.pop("scipy.stats", None)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    motif = np.sin(np.linspace(0, np.pi, 20))
    segments = []
    labels = []
    for _ in range(200):
        if np.random.rand() < 0.5:
            segments.append(motif + np.random.normal(0, 0.05, size=len(motif)))
            labels.append(np.ones(len(motif)))
        else:
            segments.append(np.random.normal(0, 1, size=len(motif)))
            labels.append(np.zeros(len(motif)))
    series = np.concatenate(segments)
    y = np.concatenate(labels)[len(motif) - 1 :]
    dtw_df = compute(series, window=len(motif), motifs=[motif])
    dist = dtw_df["dtw_dist_0"].to_numpy()[len(motif) - 1 :]
    X_base = np.random.normal(size=len(dist)).reshape(-1, 1)
    X_dtw = np.column_stack([X_base, dist])
    clf = LogisticRegression(solver="liblinear")
    base_score = cross_val_score(clf, X_base, y, cv=3, scoring="f1").mean()
    dtw_score = cross_val_score(clf, X_dtw, y, cv=3, scoring="f1").mean()
    assert dtw_score > base_score


def test_add_dtw_features_pairwise():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Timestamp": idx.tolist() * 3,
            "Symbol": ["AAA"] * 5 + ["BBB"] * 5 + ["CCC"] * 5,
            "mid": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        }
    )
    df["return"] = df.groupby("Symbol")["mid"].pct_change().fillna(0.0)

    out = add_dtw_features(df, window=3, whitelist=["AAA", "BBB", "CCC"])
    assert "dtw_AAA_BBB" in out.columns
    assert "dtw_AAA_CCC" in out.columns

    last_ts = idx[-1]
    last_row = out[(out["Symbol"] == "AAA") & (out["Timestamp"] == last_ts)]
    assert last_row["dtw_AAA_BBB"].iloc[0] == pytest.approx(0.0)
    assert last_row["dtw_AAA_CCC"].iloc[0] > 0.0
