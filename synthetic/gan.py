"""A lightweight TimeGAN-inspired generator implemented with PyTorch."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class _GANConfig:
    batch_size: int = 128
    rnn_hidden_dim: int = 24
    latent_dim: int = 8
    learning_rate: float = 5e-4


class _Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.lstm(z)
        out = self.proj(out)
        return self.tanh(out)


class _Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        _, (hidden, _) = self.lstm(x)
        logits = self.fc(hidden[-1])
        return logits.squeeze(-1)


class TimeGAN:
    """A minimal TimeGAN replacement compatible with the training script.

    The implementation focuses on API compatibility rather than reproducing the
    original research model.  It trains a simple recurrent GAN that generates
    sequences with the same shape as the training data.  While lightweight, the
    model provides a practical alternative when the ``ydata-synthetic`` package
    is unavailable on recent Python versions.
    """

    def __init__(
        self,
        model_params: dict[str, float | int] | None,
        *,
        seq_len: int,
        n_seq: int,
        device: str | torch.device | None = None,
    ) -> None:
        cfg = _GANConfig()
        if model_params:
            for field in ("batch_size", "rnn_hidden_dim", "latent_dim", "learning_rate"):
                if field in model_params:
                    value = model_params[field]
                    if field == "batch_size" or field.endswith("_dim"):
                        setattr(cfg, field, int(value))
                    else:
                        setattr(cfg, field, float(value))

        self.seq_len = seq_len
        self.n_seq = n_seq
        self.config = cfg
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.generator = _Generator(cfg.latent_dim, cfg.rnn_hidden_dim, n_seq).to(self.device)
        self.discriminator = _Discriminator(n_seq, cfg.rnn_hidden_dim).to(self.device)
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=cfg.learning_rate)
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def _noise(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.seq_len, self.config.latent_dim, device=self.device)
        return z

    def train(self, data: np.ndarray, epochs: int) -> None:
        if data.ndim != 3:
            raise ValueError("Input data must have shape (n_samples, seq_len, n_features)")
        if data.shape[1] != self.seq_len or data.shape[2] != self.n_seq:
            raise ValueError("Input sequence dimensions do not match the model configuration")

        tensor_data = torch.as_tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)

        real_label = 0.9  # label smoothing for stability
        fake_label = 0.1

        self.generator.train()
        self.discriminator.train()

        for _ in range(max(epochs, 1)):
            for (real_batch,) in loader:
                real_batch = real_batch.to(self.device)
                batch_size = real_batch.size(0)

                # Train discriminator
                self.optim_d.zero_grad()
                noise = self._noise(batch_size)
                fake_batch = self.generator(noise).detach()
                logits_real = self.discriminator(real_batch)
                logits_fake = self.discriminator(fake_batch)
                labels_real = torch.full_like(logits_real, real_label, device=self.device)
                labels_fake = torch.full_like(logits_fake, fake_label, device=self.device)
                loss_real = self.criterion(logits_real, labels_real)
                loss_fake = self.criterion(logits_fake, labels_fake)
                loss_d = loss_real + loss_fake
                loss_d.backward()
                self.optim_d.step()

                # Train generator
                self.optim_g.zero_grad()
                noise = self._noise(batch_size)
                generated = self.generator(noise)
                logits = self.discriminator(generated)
                target = torch.full_like(logits, real_label, device=self.device)
                loss_g = self.criterion(logits, target)
                loss_g.backward()
                self.optim_g.step()

    def sample(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        self.generator.eval()
        samples = []
        remaining = n_samples
        with torch.no_grad():
            while remaining > 0:
                batch_size = min(self.config.batch_size, remaining)
                noise = self._noise(batch_size)
                generated = self.generator(noise).cpu().numpy()
                samples.append(generated)
                remaining -= batch_size
        self.generator.train()
        return np.concatenate(samples, axis=0)[:n_samples]
