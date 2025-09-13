import numpy as np
import torch
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from analysis.market_simulator import AdversarialMarketSimulator


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        return self.linear(prices.unsqueeze(-1)).squeeze(-1)

    def loss(self, prices: torch.Tensor) -> torch.Tensor:
        pred = self.forward(prices)
        target = torch.ones_like(pred)
        return -((pred - target) ** 2).mean()


def train(policy: ToyPolicy, data: np.ndarray, epochs: int = 50) -> None:
    opt = torch.optim.SGD(policy.parameters(), lr=0.05)
    tensor = torch.tensor(data, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -policy.loss(tensor)
        loss.backward()
        opt.step()


def train_adv(policy: ToyPolicy, prices: np.ndarray, epochs: int = 50) -> None:
    opt = torch.optim.SGD(policy.parameters(), lr=0.05)
    sim = AdversarialMarketSimulator(seq_len=len(prices), eps=0.5)
    base = np.array(prices, dtype=float)
    for _ in range(epochs):
        adv = sim.perturb(base, policy)
        tensor = torch.tensor(adv, dtype=torch.float32)
        opt.zero_grad()
        loss = -policy.loss(tensor)
        loss.backward()
        opt.step()


def test_adversarial_sequences_reduce_overfitting():
    torch.manual_seed(0)
    prices = np.ones(32, dtype=float)

    base = ToyPolicy()
    train(base, prices)

    adv_policy = ToyPolicy()
    train_adv(adv_policy, prices)

    stress = np.zeros_like(prices)
    prices_t = torch.tensor(prices, dtype=torch.float32)
    stress_t = torch.tensor(stress, dtype=torch.float32)
    base_gap = abs(base.loss(prices_t).item() - base.loss(stress_t).item())
    adv_gap = abs(adv_policy.loss(prices_t).item() - adv_policy.loss(stress_t).item())

    assert adv_gap < base_gap
