"""Change point detection for feature distributions.

This module provides utilities to detect structural breaks in features. It
relies on the :mod:`ruptures` library when available and otherwise falls back
to a simple mean-shift heuristic. Detected breakpoints and summary statistics
are stored under ``reports/change_points`` which can be surfaced on dashboards
or used to trigger model retraining.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import ruptures as rpt
except Exception:  # pragma: no cover - ``ruptures`` may not be installed
    rpt = None  # type: ignore


class ChangePointDetector:
    """Detect change points in numeric columns of a DataFrame."""

    def __init__(self, penalty: float = 5.0, threshold: float = 1.0) -> None:
        """Create a detector.

        Parameters
        ----------
        penalty: float
            Penalty value forwarded to :mod:`ruptures` when available. Higher
            values reduce the number of detected breakpoints.
        threshold: float
            Minimum absolute mean difference used by the fallback heuristic
            when :mod:`ruptures` is unavailable.
        """

        self.penalty = penalty
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def detect(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Locate change points for each numeric column.

        Parameters
        ----------
        df: pd.DataFrame
            Input data.

        Returns
        -------
        Dict[str, List[int]]
            Mapping of column names to a list of breakpoint indices.
        """

        results: Dict[str, List[int]] = {}
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            series = numeric[col].dropna().values
            if len(series) < 2:
                results[col] = []
                continue

            if rpt is not None:
                # use PELT algorithm which scales to long time series
                model = rpt.Pelt(model="rbf").fit(series)
                bps = model.predict(pen=self.penalty)
                # ``ruptures`` includes length of series as last breakpoint
                bps = [bp for bp in bps if bp < len(series)]
            else:  # fallback to simple mean-shift detection
                mid = len(series) // 2
                mean1 = series[:mid].mean()
                mean2 = series[mid:].mean()
                bps = [mid] if abs(mean2 - mean1) > self.threshold else []
            results[col] = [int(bp) for bp in bps]
        return results

    # ------------------------------------------------------------------
    def record(
        self, df: pd.DataFrame, out_dir: Path = Path("reports/change_points")
    ) -> Dict[str, List[int]]:
        """Detect and persist change points to ``out_dir``.

        A file named ``YYYY-MM-DD.json`` and ``latest.json`` are written when
        any change points are detected. A ``retrain.flag`` file is also created
        to allow external processes to pause/queue model retraining.

        Parameters
        ----------
        df: pd.DataFrame
            Input data to analyse.
        out_dir: Path
            Directory where reports are written.

        Returns
        -------
        Dict[str, List[int]]
            Breakpoints detected for each column.
        """

        cps = self.detect(df)
        if not any(cps.values()):
            return cps

        out_dir.mkdir(parents=True, exist_ok=True)

        records: Dict[str, List[dict]] = {}
        for col, points in cps.items():
            if not points:
                continue
            series = df[col].dropna().values
            recs: List[dict] = []
            for bp in points:
                recs.append(
                    {
                        "index": int(bp),
                        "mean_before": float(series[:bp].mean()),
                        "mean_after": float(series[bp:].mean()),
                    }
                )
            records[col] = recs

        ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        with open(out_dir / f"{ts}.json", "w") as fh:
            json.dump(records, fh)
        with open(out_dir / "latest.json", "w") as fh:
            json.dump(records, fh)

        # flag to pause/queue retraining tasks
        (out_dir / "retrain.flag").write_text("change point detected")

        return cps


__all__ = ["ChangePointDetector"]

