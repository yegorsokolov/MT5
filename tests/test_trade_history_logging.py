from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import log_utils


def test_log_trade_history_dedup(monkeypatch, tmp_path):
    history = tmp_path / "trade_history.parquet"
    monkeypatch.setattr(log_utils, "TRADE_HISTORY", history)
    rec = {
        "order_id": 1,
        "Symbol": "EURUSD",
        "Timestamp": "2023-01-01T00:00:00",
        "side": "BUY",
    }
    log_utils.log_trade_history(rec)
    log_utils.log_trade_history(rec)
    df = pd.read_parquet(history)
    assert len(df) == 1
    assert df.iloc[0]["order_id"] == 1
