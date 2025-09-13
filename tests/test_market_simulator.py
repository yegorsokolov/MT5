import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.market_simulator import AdversarialMarketSimulator


class ConstantPolicy(torch.nn.Module):
    """Simple policy predicting a constant return."""

    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def loss(self, prices: torch.Tensor) -> torch.Tensor:
        returns = prices[1:] - prices[:-1]
        pred = self.param.expand_as(returns)
        return torch.nn.functional.mse_loss(pred, returns)


def _train(policy: ConstantPolicy, prices: np.ndarray, epochs: int = 200) -> None:
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    prices_t = torch.tensor(prices, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = policy.loss(prices_t)
        loss.backward()
        opt.step()


def _eval_loss(policy: ConstantPolicy, prices: np.ndarray) -> float:
    with torch.no_grad():
        return float(policy.loss(torch.tensor(prices, dtype=torch.float32)).item())


def test_adversarial_sequences_reduce_overfitting() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # training data exhibits a strong upward trend
    train_prices = np.arange(0, 11, dtype=np.float32)
    # test data moves in the opposite direction
    test_prices = np.arange(10, -1, -1, dtype=np.float32)

    # baseline training on original data
    base_policy = ConstantPolicy()
    _train(base_policy, train_prices)
    base_loss = _eval_loss(base_policy, test_prices)

    # adversarial training alternates policy and simulator updates
    adv_policy = ConstantPolicy()
    sim = AdversarialMarketSimulator(seq_len=len(train_prices), eps=2.0)
    opt = torch.optim.SGD(adv_policy.parameters(), lr=0.1)
    for _ in range(200):
        # policy update on real data
        opt.zero_grad()
        loss = adv_policy.loss(torch.tensor(train_prices, dtype=torch.float32))
        loss.backward()
        opt.step()

        # adversary generates challenging examples
        adv_prices = sim.perturb(train_prices, adv_policy, steps=3)
        opt.zero_grad()
        loss = adv_policy.loss(torch.tensor(adv_prices, dtype=torch.float32))
        loss.backward()
        opt.step()

    adv_loss = _eval_loss(adv_policy, test_prices)

    assert adv_loss < base_loss
