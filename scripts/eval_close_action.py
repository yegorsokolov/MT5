import numpy as np
import pandas as pd
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
metrics_stub = types.SimpleNamespace(record_metric=lambda *a, **k: None, TS_PATH="")
sys.modules.setdefault("analytics", types.SimpleNamespace(metrics_store=metrics_stub))
sys.modules.setdefault("analytics.metrics_store", metrics_stub)
from rl.trading_env import TradingEnv


def run_episode(use_close: bool):
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=5, freq="H"),
            "Symbol": ["A"] * 5,
            "mid": np.linspace(100, 104, 5),
            "return": np.zeros(5),
        }
    )
    env = TradingEnv(df, ["return"], transaction_cost=0.0, risk_penalty=0.0, exit_penalty=0.0)
    env.reset()
    hold_steps = 0
    total_ret = 0.0
    for t in range(len(df) - 1):
        if use_close and t == (len(df) // 2):
            action = [0.0, 1.0]
        else:
            action = [1.0]
        _, _, done, info = env.step(action)
        hold_steps += int(env.positions[0] != 0.0)
        total_ret += info["portfolio_return"]
        if done:
            break
    return hold_steps, total_ret


def main() -> None:
    hold_no_close, ret_no_close = run_episode(False)
    hold_close, ret_close = run_episode(True)
    print(f"No close action: hold_steps={hold_no_close}, return={ret_no_close:.4f}")
    print(f"With close action: hold_steps={hold_close}, return={ret_close:.4f}")


if __name__ == "__main__":
    main()
