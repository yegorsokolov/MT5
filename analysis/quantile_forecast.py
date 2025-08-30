from __future__ import annotations

"""Utilities for logging quantile forecast coverage."""

from pathlib import Path
import os
import csv
from typing import Optional


def log_quantile_forecast(
    timestamp: str,
    symbol: str,
    alpha: float,
    var: Optional[float],
    realised: Optional[float],
) -> None:
    """Append a quantile forecast record to ``reports/quantile_forecasts``.

    Parameters
    ----------
    timestamp:
        Timestamp of the forecast.
    symbol:
        Asset identifier.
    alpha:
        Tail probability of the VaR forecast.
    var:
        Predicted Value-at-Risk.
    realised:
        Realised return. If provided, a coverage indicator is logged showing
        whether the realised return exceeded the predicted VaR.
    """

    report_dir = Path(
        os.getenv(
            "QUANTILE_FORECAST_REPORT_PATH",
            Path(__file__).resolve().parents[1] / "reports" / "quantile_forecasts",
        )
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    file = report_dir / "quantile_forecasts.csv"
    write_header = not file.exists()
    coverage = ""
    if realised is not None and var is not None:
        coverage = 1.0 if realised >= var else 0.0
    with file.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Timestamp", "Symbol", "alpha", "var", "realised", "coverage", "target"])
        writer.writerow([
            timestamp,
            symbol,
            alpha,
            "" if var is None else var,
            "" if realised is None else realised,
            coverage,
            1 - alpha,
        ])


__all__ = ["log_quantile_forecast"]
