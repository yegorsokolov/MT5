import importlib.util
import sys
from pathlib import Path

import numpy as np

# Import the PER module directly to avoid heavy rl package side effects
spec = importlib.util.spec_from_file_location(
    "prioritized_replay", Path(__file__).resolve().parents[1] / "rl" / "per" / "prioritized_replay.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # type: ignore[arg-type]
PrioritizedReplayBuffer = module.PrioritizedReplayBuffer


def test_sampling_bias():
    buf = PrioritizedReplayBuffer(4, alpha=0.6, beta=0.4)
    priorities = [1, 1, 1, 10]
    for i, p in enumerate(priorities):
        buf.add(i, priority=p)

    counts = np.zeros(len(priorities))
    for _ in range(1000):
        _, idxs, _ = buf.sample(1, with_weights=True)
        counts[idxs[0]] += 1

    assert counts[3] > counts[0] * 2  # high-priority item should be sampled more


def test_weight_correction():
    buf = PrioritizedReplayBuffer(4, alpha=1.0, beta=1.0)
    priorities = [1, 2, 3, 4]
    for i, p in enumerate(priorities):
        buf.add(i, priority=p)

    _, idxs, weights = buf.sample(4, with_weights=True)
    probs = np.array(priorities, dtype=float)
    probs = probs / probs.sum()
    expected = (len(priorities) * probs[idxs]) ** -1
    expected /= expected.max()

    assert np.allclose(weights, expected)
