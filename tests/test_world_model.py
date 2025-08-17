import sys
from pathlib import Path
import importlib.util

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

spec = importlib.util.spec_from_file_location(
    "world_model", repo_root / "rl" / "world_model.py"
)
world_model = importlib.util.module_from_spec(spec)
sys.modules["world_model"] = world_model
spec.loader.exec_module(world_model)
WorldModel = world_model.WorldModel
WorldModelEnv = world_model.WorldModelEnv
Transition = world_model.Transition


class SimpleEnv:
    """Deterministic 1D environment used for world model tests."""

    def __init__(self) -> None:
        self.state = 0.0

    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=float)

    def step(self, action):
        a = float(action[0] if isinstance(action, (list, np.ndarray)) else action)
        next_state = np.array([a], dtype=float)
        reward = -(a - 0.5) ** 2
        self.state = a
        done = True
        return next_state, reward, done, {}


def _collect_transitions(env, n=20):
    transitions = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        state = env.reset()
        action = rng.uniform(-1, 1, size=(1,))
        next_state, reward, _, _ = env.step(action)
        transitions.append(Transition(state.tolist(), action.tolist(), next_state.tolist(), float(reward)))
    return transitions


def test_world_model_enables_faster_convergence():
    env = SimpleEnv()
    transitions = _collect_transitions(env, n=30)
    wm = WorldModel(1, 1)
    wm.train(transitions)

    # Plan inside the world model via grid search
    actions = np.linspace(-1, 1, 101).reshape(-1, 1)
    preds = [wm.predict([0.0], a)[1] for a in actions]
    best_action = actions[int(np.argmax(preds))]
    env.reset()
    _, wm_reward, _, _ = env.step(best_action)

    # Pure model-free random search with limited interactions
    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(5):
        env.reset()
        a = rng.uniform(-1, 1, size=(1,))
        _, r, _, _ = env.step(a)
        rewards.append(r)
    baseline_reward = max(rewards)

    assert wm_reward > baseline_reward
    assert wm_reward >= -1e-3  # near optimal reward (0)
