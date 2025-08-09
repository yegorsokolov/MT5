from __future__ import annotations

"""Generate monthly performance summaries for auditing purposes."""

from pathlib import Path
import argparse

import pandas as pd

from analytics.metrics_store import MetricsStore


def generate_report(output: str = "reports/monthly_performance.csv") -> Path:
    """Create a monthly performance report from stored daily metrics.

    Parameters
    ----------
    output: str
        Destination CSV file.  Parent directories are created as needed.
    """

    store = MetricsStore()
    df = store.load()
    if df.empty:
        raise ValueError("No metrics available to build report")

    monthly = df.resample("M").agg({
        "return": "sum",
        "sharpe": "mean",
        "drawdown": "min",
    })
    monthly.index = monthly.index.strftime("%Y-%m")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out_path)
    return out_path


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/monthly_performance.csv")
    args = parser.parse_args()
    path = generate_report(args.output)
    print(f"Saved report to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
