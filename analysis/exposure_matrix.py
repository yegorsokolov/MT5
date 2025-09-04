from __future__ import annotations

"""Maintain rolling correlation and correlation-weighted exposure matrix."""

from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ExposureMatrix:
    """Track correlations between instruments and derive exposure matrix.

    Parameters
    ----------
    window: int, optional
        Number of periods for the rolling correlation estimation.  Defaults to
        20.
    """

    window: int = 20
    corr: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        self._returns: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    # ------------------------------------------------------------------
    def update_returns(self, returns: Dict[str, float]) -> None:
        """Update rolling correlation matrix from symbol ``returns``."""

        for sym, ret in returns.items():
            self._returns[sym].append(ret)
        if not self._returns:
            return
        df = pd.DataFrame(self._returns)
        if len(df) < 2:
            return
        self.corr = df.corr().fillna(0.0)

    # ------------------------------------------------------------------
    def weighted_exposures(self, exposures: Dict[str, float]) -> pd.DataFrame:
        """Return correlation-weighted exposure matrix.

        The resulting DataFrame has symbols on both axes and each cell contains
        ``exposure_i * exposure_j * corr_ij``.  Missing correlations default to
        zero while diagonal elements are always one.
        """

        if not exposures:
            return pd.DataFrame()
        symbols = list(exposures)
        vec = np.array([exposures[s] for s in symbols], dtype=float)
        if self.corr.empty:
            corr = np.eye(len(symbols))
        else:
            corr = (
                self.corr.reindex(index=symbols, columns=symbols).fillna(0.0).to_numpy()
            )
            np.fill_diagonal(corr, 1.0)
        mat = np.outer(vec, vec) * corr
        return pd.DataFrame(mat, index=symbols, columns=symbols)

    # ------------------------------------------------------------------
    def snapshot(
        self, exposures: Dict[str, float], path: str = "reports/exposure_matrix"
    ) -> None:
        """Persist current weighted exposures to ``path``.

        Both a timestamped CSV and a ``latest.json`` file are written for the
        dashboard to consume.  Errors during persistence are silently ignored.
        """

        try:
            matrix = self.weighted_exposures(exposures)
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
            matrix.to_csv(p / f"{ts}.csv")
            matrix.to_json(p / "latest.json")
        except Exception:
            pass


__all__ = ["ExposureMatrix"]
