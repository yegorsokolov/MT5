"""Compare single-agent and multi-agent RL on synthetic correlated markets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from train_rl import PositionClosePolicy
from rl.trading_env import TradingEnv
from rl.multi_agent_env import MultiAgentTradingEnv


def _make_synthetic(n: int = 200) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=n, freq="min")
    cov = np.array([[0.001, 0.0008], [0.0008, 0.001]])
    rets = np.random.multivariate_normal([0, 0], cov, size=n)
    prices = 100 * np.cumprod(1 + rets, axis=0)
    dfs = []
    for i, sym in enumerate(["A", "B"]):
        dfs.append(
            pd.DataFrame(
                {
                    "Timestamp": ts,
                    "Symbol": sym,
                    "mid": prices[:, i],
                    "return": rets[:, i],
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


def train_single(df: pd.DataFrame):
    env = TradingEnv(df, ["return"], max_position=1.0, transaction_cost=0.0, risk_penalty=0.0)
    model = PPO(PositionClosePolicy, env, verbose=0, seed=0, learning_rate=3e-4)
    model.learn(total_timesteps=1000)
    return env, model


def train_multi(df: pd.DataFrame):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    env_kwargs = dict(
        df=df,
        features=["return"],
        max_position=1.0,
        transaction_cost=0.0,
        risk_penalty=0.0,
        var_window=30,
    )

    def env_creator(env_config=None):
        return MultiAgentTradingEnv(**env_kwargs)

    temp_env = env_creator()
    policies = {
        sym: (None, temp_env.observation_space[sym], temp_env.action_space[sym], {})
        for sym in temp_env.agents
    }
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    config = (
        PPOConfig()
        .environment(env_creator, disable_env_checking=True)
        .rollouts(num_rollout_workers=0)
        .training(model={"vf_share_layers": True})
        .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, *a, **k: agent_id)
    )
    algo = config.build()
    for _ in range(5):
        algo.train()
    ray.shutdown()
    env = env_creator()
    return env, algo


def eval_single(env: TradingEnv, model) -> float:
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
    return float((1 + np.array(env.portfolio_returns)).prod() - 1)


def eval_multi(env: MultiAgentTradingEnv, algo) -> float:
    obs = env.reset()
    done = False
    while not done:
        actions = {
            aid: algo.get_policy(aid).compute_single_action(ob)[0] for aid, ob in obs.items()
        }
        obs, rewards, dones, infos = env.step(actions)
        done = dones["__all__"]
    return float((1 + np.array(env.portfolio_returns)).prod() - 1)


def main() -> None:
    df = _make_synthetic()
    single_env, single_model = train_single(df)
    multi_env, multi_algo = train_multi(df)
    single_ret = eval_single(single_env, single_model)
    multi_ret = eval_multi(multi_env, multi_algo)
    print(f"Single-agent return: {single_ret:.4f}")
    print(f"Multi-agent return: {multi_ret:.4f}")


if __name__ == "__main__":
    main()
