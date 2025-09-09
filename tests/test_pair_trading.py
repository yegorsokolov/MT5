import numpy as np
import pandas as pd
import sys
from pathlib import Path
import importlib.util

# Ensure real SciPy is available (conftest stubs a minimal version)
for mod in ["scipy", "scipy.stats"]:
    sys.modules.pop(mod, None)

_pair_path = Path(__file__).resolve().parents[1] / "strategy" / "pair_trading.py"
spec = importlib.util.spec_from_file_location("pair_trading", _pair_path)
pair_trading = importlib.util.module_from_spec(spec)
sys.modules["pair_trading"] = pair_trading
spec.loader.exec_module(pair_trading)
find_cointegrated_pairs = pair_trading.find_cointegrated_pairs
generate_signals = pair_trading.generate_signals


def build_df() -> pd.DataFrame:
    """Construct synthetic cointegrated price series."""
    rng = np.random.default_rng(0)
    n = 300
    ts = pd.date_range("2020-01-01", periods=n, freq="min")
    noise = rng.normal(0, 0.1, size=n)
    price_a = noise.cumsum() + 100
    price_b = 2 * price_a + rng.normal(0, 0.1, size=n)
    price_b[250:260] += 2  # temporary divergence
    df_a = pd.DataFrame({"Timestamp": ts, "Symbol": "A", "Bid": price_a})
    df_b = pd.DataFrame({"Timestamp": ts, "Symbol": "B", "Bid": price_b})
    return pd.concat([df_a, df_b], ignore_index=True)


def test_signal_generation_and_pnl():
    df = build_df()
    pairs = find_cointegrated_pairs(df, significance=0.2)
    assert any({"A", "B"} == {p[0], p[1]} for p in pairs)
    result = generate_signals(df, window=5, entry_z=1.0, exit_z=0.0, significance=0.2)
    out = result.df
    assert "pair_z_A_B" in out.columns
    # Ensure at least one trade signal was generated
    assert out["pair_long"].sum() > 0 or out["pair_short"].sum() > 0
    # PnL should be non-zero when trades occur
    assert abs(out["pair_pnl"].sum()) > 0
