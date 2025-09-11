import sys
from pathlib import Path
import importlib
import importlib.util
import numpy as np
import pandas as pd

# Ensure repository root on path to avoid shadowing by installed packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_coin_spec = importlib.util.spec_from_file_location(
    "features.cointegration", ROOT / "features" / "cointegration.py"
)
cointegration = importlib.util.module_from_spec(_coin_spec)
sys.modules["features.cointegration"] = cointegration
assert _coin_spec.loader is not None
_coin_spec.loader.exec_module(cointegration)

_pair_spec = importlib.util.spec_from_file_location(
    "strategy.pair_trading", ROOT / "strategy" / "pair_trading.py"
)
pair_trading = importlib.util.module_from_spec(_pair_spec)
sys.modules["strategy.pair_trading"] = pair_trading
assert _pair_spec.loader is not None
_pair_spec.loader.exec_module(pair_trading)


def build_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    ts = pd.date_range("2020-01-01", periods=n, freq="min")
    noise = rng.normal(0, 0.1, size=n)
    price_a = noise.cumsum() + 50
    price_b = 2 * price_a + rng.normal(0, 0.1, size=n)
    price_b[150:160] += 1.5
    df_a = pd.DataFrame({"Timestamp": ts, "Symbol": "A", "Bid": price_a})
    df_b = pd.DataFrame({"Timestamp": ts, "Symbol": "B", "Bid": price_b})
    return pd.concat([df_a, df_b], ignore_index=True)


def test_feature_and_strategy_integration():
    df = build_df()
    feats = cointegration.compute(df, pairs=[("A", "B")], window=5)
    assert "pair_z_A_B" in feats.columns
    result = pair_trading.generate_signals(feats, entry_z=0.1, exit_z=0.0)
    out = result.df
    assert out["pair_long"].sum() > 0 or out["pair_short"].sum() > 0
    assert abs(out["pair_pnl"].sum()) > 0
