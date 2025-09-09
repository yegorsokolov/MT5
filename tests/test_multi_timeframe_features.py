import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path

# Load module directly to avoid heavy package imports
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "multi_timeframe", ROOT / "features" / "multi_timeframe.py"
)
multi_timeframe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_timeframe)


def test_resampled_indicators():
    ts = pd.date_range("2024-01-01", periods=8 * 60, freq="min")
    mid = np.sin(np.linspace(0, 10, len(ts)))
    df = pd.DataFrame({"Timestamp": ts, "mid": mid})

    out = multi_timeframe.compute(df, timeframes=("1h", "4h"), ma_window=1, rsi_window=2)

    base = df.set_index(ts)["mid"]

    res1h = base.resample("1h").last()
    exp_ma1h = res1h.rolling(1).mean().reindex(base.index, method="ffill")
    exp_rsi1h = multi_timeframe._rsi(res1h, 2).reindex(base.index, method="ffill")
    np.testing.assert_allclose(out["ma_1h"], exp_ma1h.reset_index(drop=True))
    np.testing.assert_allclose(
        out["rsi_1h"], exp_rsi1h.reset_index(drop=True), equal_nan=True
    )

    res4h = base.resample("4h").last()
    exp_ma4h = res4h.rolling(1).mean().reindex(base.index, method="ffill")
    exp_rsi4h = multi_timeframe._rsi(res4h, 2).reindex(base.index, method="ffill")
    np.testing.assert_allclose(out["ma_4h"], exp_ma4h.reset_index(drop=True))
    np.testing.assert_allclose(
        out["rsi_4h"], exp_rsi4h.reset_index(drop=True), equal_nan=True
    )
