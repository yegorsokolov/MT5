import numpy as np
import pandas as pd
import sys
import types

sys.modules['analytics.metrics_store'] = types.SimpleNamespace(
    record_metric=lambda *a, **k: None, TS_PATH=""
)

from rl.trading_env import TradingEnv


def _make_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(0)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="min")
    news = np.random.choice([-1, 1], size=n)
    returns = np.zeros(n)
    returns[1:] = news[:-1] * 0.01
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Symbol": ["A"] * n,
            "mid": prices,
            "return": returns,
            "news_movement_score": news,
        }
    )
    return df


def _sharpe(returns: np.ndarray) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))


def _run_policy(env: TradingEnv, policy_fn) -> np.ndarray:
    obs = env.reset()
    done = False
    while not done:
        action = policy_fn(obs)
        obs, _, done, _ = env.step(action)
    return np.array(env.portfolio_returns)


def test_news_features_improve_sharpe():
    df = _make_df()

    env_news = TradingEnv(df.copy(), ["return"], news_window=1)

    def news_policy(obs: np.ndarray) -> np.ndarray:
        return np.array([np.sign(obs[-1])], dtype=np.float32)

    returns_news = _run_policy(env_news, news_policy)
    sharpe_news = _sharpe(returns_news)

    env_base = TradingEnv(df.copy(), ["return"])

    def base_policy(obs: np.ndarray) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    returns_base = _run_policy(env_base, base_policy)
    sharpe_base = _sharpe(returns_base)

    assert sharpe_news >= sharpe_base

