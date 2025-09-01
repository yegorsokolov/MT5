import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _load_inverse_reward():
    rl_stub = types.ModuleType("rl")
    offline_stub = types.ModuleType("rl.offline_dataset")
    offline_stub.OfflineDataset = object
    sys.modules["rl"] = rl_stub
    sys.modules["rl.offline_dataset"] = offline_stub
    rl_stub.offline_dataset = offline_stub
    spec = importlib.util.spec_from_file_location(
        "rl.inverse_reward", repo_root / "rl" / "inverse_reward.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rl.inverse_reward"] = mod
    spec.loader.exec_module(mod)
    sys.modules.pop("rl")
    sys.modules.pop("rl.offline_dataset")
    return mod


def test_maxent_irl_improves_likelihood():
    inv = _load_inverse_reward()
    features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    actions = np.array([1, 1, 0, 1])
    theta = inv.maxent_irl(features, actions, lr=0.5, iterations=200)
    hand_theta = np.zeros(2)

    def log_lik(th):
        logits = features @ th
        probs = 1 / (1 + np.exp(-logits))
        return np.sum(actions * np.log(probs) + (1 - actions) * np.log(1 - probs))

    assert log_lik(theta) > log_lik(hand_theta)
