import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis import reprocess_trades

reprocess_trades_module = reprocess_trades
reprocess_trades = reprocess_trades_module.reprocess_trades


def test_reprocess_generates_report(tmp_path):
    history = tmp_path / "trade_history.parquet"
    models_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    models_dir.mkdir()
    df = pd.DataFrame(
        {
            "order_id": [1],
            "Timestamp": [pd.Timestamp("2023-01-01")],
            "Symbol": ["EURUSD"],
            "side": ["BUY"],
            "feature1": [0.2],
            "prob": [0.6],
        }
    )
    df.to_parquet(history)
    model = LogisticRegression().fit([[0.1], [0.5]], [0, 1])
    joblib.dump(model, models_dir / "model.joblib")
    reprocess_trades(history_path=history, models_dir=models_dir, report_dir=report_dir)
    files = list(report_dir.glob("*.parquet"))
    assert files, "report not generated"
    out = pd.read_parquet(files[0])
    assert "new_prob" in out.columns


def test_hold_duration_report(tmp_path, monkeypatch):
    history = tmp_path / "trade_history.parquet"
    models_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    hold_dir = tmp_path / "hold_duration"
    monkeypatch.setattr(reprocess_trades_module, "HOLD_REPORT_DIR", hold_dir)
    models_dir.mkdir()
    df = pd.DataFrame(
        {
            "order_id": [1],
            "Timestamp": [pd.Timestamp("2024-01-01")],
            "Symbol": ["EURUSD"],
            "side": ["BUY"],
            "volume": [1.0],
            "entry_time": [pd.Timestamp("2024-01-01 09:00")],
            "exit_time": [pd.Timestamp("2024-01-01 09:30")],
            "entry_price": [100],
            "exit_price": [110],
            "feature1": [0.2],
        }
    )
    df.to_parquet(history)
    model = LogisticRegression().fit([[0.1], [0.5]], [0, 1])
    joblib.dump(model, models_dir / "model.joblib")
    reprocess_trades(history_path=history, models_dir=models_dir, report_dir=report_dir)
    hold_report = hold_dir / "pnl_by_duration.csv"
    assert hold_report.exists(), "hold duration report not generated"
    hdf = pd.read_csv(hold_report)
    assert 30 in hdf["duration_min"].tolist()
