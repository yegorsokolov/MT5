import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.multi_agent_env import MultiAgentTradingEnv


def test_multi_agent_env_step():
    n = 5
    ts = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "Timestamp": np.tile(ts, 2),
            "Symbol": ["A"] * n + ["B"] * n,
            "mid": 1.0,
            "return": 0.0,
        }
    )
    env = MultiAgentTradingEnv(df, ["return"])
    obs = env.reset()
    assert set(obs.keys()) == set(env.agents)
    actions = {aid: 0.0 for aid in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    assert set(rewards.keys()) == set(env.agents)
    assert "__all__" in dones
