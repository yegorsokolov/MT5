import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from models.slimmable_network import SlimmableNetwork


def test_set_width_and_export():
    net = SlimmableNetwork(input_dim=4, hidden_dim=8, width_multipliers=[0.25, 0.5, 1.0])
    # forward at half width
    net.set_width(0.5)
    x = torch.randn(2, 4)
    out = net(x)
    assert out.shape == (2, 1)
    # export slices
    slices = net.export_slices()
    assert 0.5 in slices and 1.0 in slices
    # check slice sizes
    half = slices[0.5]["fc1.weight"].shape[0]
    assert half == int(8 * 0.5)
