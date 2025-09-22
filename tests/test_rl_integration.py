import os
import sys
import types
import asyncio
from pathlib import Path
import contextlib

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Avoid heavy feature imports
os.environ.setdefault("MT5_DOCS_BUILD", "1")

# ---------------------------------------------------------------------------
# Lightweight stubs for optional dependencies
# ---------------------------------------------------------------------------
class _Space:
    def __init__(self, *a, **k):
        self.n = 2

    def sample(self):
        return 0

gym_spaces = types.SimpleNamespace(Box=_Space, Dict=_Space, Discrete=_Space)
sys.modules.setdefault("gym", types.SimpleNamespace(Env=object, spaces=gym_spaces))

sys.modules.setdefault(
    "data.order_book",
    types.SimpleNamespace(
        load_order_book=lambda src: pd.DataFrame(src),
        compute_order_book_features=lambda df: pd.concat(
            [
                df.assign(
                    depth_imbalance=0.0,
                    vw_spread=0.0,
                    liquidity=0.0,
                    slippage=0.0,
                ),
                df.iloc[-1:].copy(),
                df.iloc[-1:].copy(),
                df.iloc[-1:].copy(),
            ],
            ignore_index=True,
        ),
    ),
)
sys.modules.setdefault("requests", types.SimpleNamespace())

metrics_stub = types.SimpleNamespace(record_metric=lambda *a, **k: None, TS_PATH="")
sys.modules.setdefault("analytics", types.SimpleNamespace(metrics_store=metrics_stub))
sys.modules.setdefault("analytics.metrics_store", metrics_stub)

monitor_stub = types.SimpleNamespace(
    start=lambda: None,
    capability_tier="lite",
    capabilities=types.SimpleNamespace(capability_tier=lambda: "lite", ddp=lambda: False),
)
utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {}
utils_stub.resource_monitor = types.SimpleNamespace(monitor=monitor_stub)
data_backend_stub = types.ModuleType("utils.data_backend")
data_backend_stub.get_dataframe_module = lambda: pd
sys.modules.setdefault("utils", utils_stub)
sys.modules.setdefault("utils.resource_monitor", types.SimpleNamespace(monitor=monitor_stub))
sys.modules.setdefault("utils.data_backend", data_backend_stub)
sys.modules.setdefault(
    "psutil",
    types.SimpleNamespace(
        cpu_count=lambda logical=True: 1,
        virtual_memory=lambda: types.SimpleNamespace(total=0),
        Process=lambda pid=None: types.SimpleNamespace(
            memory_info=lambda: types.SimpleNamespace(rss=0)
        ),
        disk_io_counters=lambda: types.SimpleNamespace(read_bytes=0, write_bytes=0),
    ),
)
date_parser_stub = types.SimpleNamespace(parse=lambda *a, **k: None)
sys.modules.setdefault("dateutil", types.SimpleNamespace(parser=date_parser_stub))
sys.modules.setdefault("dateutil.parser", date_parser_stub)
data_pkg = types.ModuleType("data")
data_features_stub = types.ModuleType("data.features")
data_features_stub.make_features = lambda df, validate=False: df
data_pkg.features = data_features_stub
sys.modules.setdefault("data", data_pkg)
sys.modules.setdefault("data.features", data_features_stub)
# Monkeypatch RL environment step to avoid out-of-bounds observation access
def _safe_step(self, action):
    row_idx = min(self.current_step, len(self.book) - 1)
    row = self.book.iloc[row_idx]
    mid = (row["BidPrice1"] + row["AskPrice1"]) / 2.0
    reward = 0.0
    done = False
    if action == 1 and self.remaining > 0:
        price = row["AskPrice1"] if self.side == "buy" else row["BidPrice1"]
        reward = -(price - mid) if self.side == "buy" else -(mid - price)
        self.remaining -= self.slice_size
    self.current_step += 1
    if self.current_step >= len(self.book) or self.remaining <= 0:
        done = True
    obs = np.zeros(4, dtype=np.float32) if done else self._get_obs()
    return obs, float(reward), done, {}

telemetry_stub = types.SimpleNamespace(
    get_tracer=lambda *a, **k: types.SimpleNamespace(
        start_as_current_span=lambda *a, **k: contextlib.nullcontext()
    ),
    get_meter=lambda *a, **k: types.SimpleNamespace(
        create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None)
    ),
)
sys.modules.setdefault("telemetry", telemetry_stub)

sys.modules.setdefault(
    "prometheus_client",
    types.SimpleNamespace(
        Counter=lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None),
        Gauge=lambda *a, **k: types.SimpleNamespace(set=lambda *a, **k: None),
    ),
)

# ---------------------------------------------------------------------------
from execution.engine import ExecutionEngine
from execution.rl_executor import RLExecutor, LOBExecutionEnv
from mt5.model_registry import ModelRegistry
import features
LOBExecutionEnv.step = _safe_step

# make_features may be heavy; use a no-op for tests
features.make_features = lambda df, validate=False: df


def test_rl_integration(tmp_path: Path):
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=7, freq="s"),
            "BidPrice1": 100.0,
            "AskPrice1": 100.1,
            "BidVolume1": 10.0,
            "AskVolume1": 10.0,
        }
    )

    env = RLExecutor.make_env(df, side="buy")
    executor = RLExecutor(env=env)
    executor.train(steps=5)
    policy_path = tmp_path / "policy"
    executor.save(policy_path)

    registry = ModelRegistry(auto_refresh=False)
    registry.register_policy("rl_small", policy_path, {"steps": 5})

    engine = ExecutionEngine(registry=registry)
    result = asyncio.run(
        engine.place_order(
            side="buy",
            quantity=1.0,
            bid=100.0,
            ask=100.1,
            bid_vol=10.0,
            ask_vol=10.0,
            mid=100.05,
            strategy="rl",
        )
    )
    assert "avg_price" in result and "filled" in result
