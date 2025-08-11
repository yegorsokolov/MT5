import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.synthetic_data import TimeSeriesGAN


def _autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def test_time_series_gan_matches_stats():
    torch.manual_seed(0)
    np.random.seed(0)
    n = 200
    data = np.random.normal(scale=0.1, size=n)

    gan = TimeSeriesGAN(seq_len=20, latent_dim=8, hidden_dim=16)
    gan.fit(data, epochs=100, batch_size=32)
    synthetic = gan.generate(n)

    orig_std = np.std(data)
    syn_std = np.std(synthetic)
    orig_ac = _autocorr(data)
    syn_ac = _autocorr(synthetic)

    assert abs(orig_std - syn_std) / orig_std < 0.3
    assert abs(orig_ac - syn_ac) < 0.2

