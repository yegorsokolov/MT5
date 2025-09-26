"""Smoke tests for the lightweight PyTorch TimeGAN replacement."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from synthetic.gan import TimeGAN


def _generate_series(n_samples: int, seq_len: int, n_features: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n_samples, seq_len, n_features)).astype(np.float32)
    trend = np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
    base += trend[None, :, None]
    return base


def test_timegan_training_and_sampling() -> None:
    seq_len = 8
    n_features = 3
    data = _generate_series(32, seq_len, n_features)

    gan = TimeGAN(
        {
            "batch_size": 8,
            "rnn_hidden_dim": 12,
            "latent_dim": 4,
            "learning_rate": 1e-3,
        },
        seq_len=seq_len,
        n_seq=n_features,
        device="cpu",
    )

    gan.train(data, epochs=1)
    samples = gan.sample(16)

    assert samples.shape == (16, seq_len, n_features)
    assert np.isfinite(samples).all()
