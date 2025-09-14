import importlib.util
import pathlib

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_stream", BASE / "rl" / "live_stream.py")
live_stream = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(live_stream)
incremental_policy_update = live_stream.incremental_policy_update


class DummyModel:
    def __init__(self):
        self.updates = 0

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False) -> None:  # noqa: D401
        self.updates += total_timesteps


class DummyEnv:
    def __init__(self):
        self.data: list[dict] = []

    def append_live_features(self, features: dict) -> None:
        self.data.append(features)


def stream(n: int):
    for i in range(n):
        yield {"mid": float(i)}


def test_incremental_policy_update():
    env = DummyEnv()
    model = DummyModel()
    incremental_policy_update(env, model, stream(3))
    assert model.updates == 3
    assert len(env.data) == 3
