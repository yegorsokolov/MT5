import sys
import types
from pathlib import Path

class _Space:
    def __init__(self, *a, **k):
        pass

class _Wrapper:
    def __init__(self, env):
        self.env = env
    def reset(self, **kwargs):  # pragma: no cover - simple proxy
        return self.env.reset(**kwargs)
    def step(self, action):  # pragma: no cover - simple proxy
        return self.env.step(action)

gym_stub = types.SimpleNamespace(Env=object, Wrapper=_Wrapper, spaces=types.SimpleNamespace(Box=_Space))
sys.modules.setdefault("gym", gym_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.risk_cvar import CVaRRewardWrapper


class DummyEnv:
    def __init__(self, rewards):
        self.rewards = rewards
        self.i = 0
        self.action_space = _Space()
        self.observation_space = _Space()
    def reset(self):
        self.i = 0
        return 0
    def step(self, action):
        r = float(self.rewards[self.i])
        self.i += 1
        done = self.i >= len(self.rewards)
        return 0, r, done, {"portfolio_return": r}


def run_episode(env):
    obs = env.reset()
    total = 0.0
    done = False
    info = {}
    while not done:
        obs, reward, done, info = env.step(0)
        total += reward
    return total, info


def test_cvar_wrapper_penalizes_rewards():
    returns = [-0.01, -0.02, -0.10, -0.10, -0.10]
    base_total, _ = run_episode(DummyEnv(returns))
    wrapped_total, info = run_episode(CVaRRewardWrapper(DummyEnv(returns), penalty=1.0, window=5, alpha=0.2))
    assert wrapped_total < base_total
    assert info["cvar"] > 0
