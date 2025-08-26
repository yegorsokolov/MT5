import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.dtw_features import compute


def test_dtw_features_improve_validation():
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
