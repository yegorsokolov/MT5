import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score

from analysis.vae_regime import VAERegime, window_features


def _make_dataset(n: int = 200, window: int = 5):
    rng = np.random.default_rng(42)
    regime0 = rng.normal(0.0, 0.1, size=(n, 2))
    regime1 = rng.normal(3.0, 0.1, size=(n, 2))
    features = np.vstack([regime0, regime1])
    labels = np.array([0] * n + [1] * n)
    windows = window_features(features, window)
    return windows, labels[window - 1 :]


def test_vae_regime_separation():
    windows, true_labels = _make_dataset()
    model = VAERegime(input_dim=windows.shape[1], latent_dim=2, hidden_dim=8)
    model.fit(windows, epochs=20, batch_size=32, lr=1e-2)
    embeddings = model.transform(windows)
    pred = model.assign_regimes(embeddings, n_clusters=2)
    acc = max(
        accuracy_score(true_labels, pred),
        accuracy_score(true_labels, 1 - pred),
    )
    assert acc > 0.8
