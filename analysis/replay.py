"""Replay historical decisions through the current model."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from analytics.metrics_store import record_metric
from news import impact_model
try:  # optional if backend not configured
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def reprocess(out_dir: Path | None = None) -> pd.DataFrame:
    """Reprocess historical trades applying the latest news impact model."""
    trade_path = Path("reports/trades.csv")
    if state_sync:
        try:
            state_sync.pull_decisions()
        except Exception:
            pass
    if not trade_path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(trade_path, parse_dates=["timestamp"])
    out_dir = Path(out_dir) if out_dir else REPORT_DIR
    adj = []
    for row in trades.itertuples():
        impact, _ = impact_model.get_impact(row.symbol, row.timestamp)
        adj.append(row.pnl * (1 + (impact or 0)))
    trades["reprocessed_pnl"] = adj
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out_dir / "reprocessed.parquet", index=False)
    summary = pd.DataFrame(
        {
            "pnl_old": [float(trades["pnl"].sum())],
            "pnl_new": [float(trades["reprocessed_pnl"].sum())],
        }
    )
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    try:
        record_metric("replay_pnl_old", summary["pnl_old"].iloc[0])
        record_metric("replay_pnl_new", summary["pnl_new"].iloc[0])
        record_metric(
            "replay_pnl_diff",
            summary["pnl_new"].iloc[0] - summary["pnl_old"].iloc[0],
        )
    except Exception:
        pass
    return summary


def main() -> None:
    res = reprocess()
    if res.empty:
        print("No trades to replay")
    else:
        print(f"Saved reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
