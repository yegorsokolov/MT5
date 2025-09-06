from __future__ import annotations

"""Streaming anomaly checks for the data pipeline.

This module exposes :func:`validate` which performs lightweight sanity checks on
incoming :class:`pandas.DataFrame` batches.  It monitors three aspects of the
pipeline:

* Tick spreads (Ask - Bid) using a running z-score detector.
* Feature distributions using running z-scores and, when available, an
  :class:`~sklearn.ensemble.IsolationForest` model.
* Prediction residuals (``target - prediction``) using a running z-score.

Metrics ``pipeline_anomaly_total`` and ``pipeline_anomaly_rate`` are updated on
each call.  When the overall anomaly rate exceeds ``alert_threshold`` an alert
is emitted.  Offending batches are written to a quarantine directory for later
inspection.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict
from collections import deque

import numpy as np
import pandas as pd

from metrics import PIPELINE_ANOMALY_TOTAL, PIPELINE_ANOMALY_RATE

try:  # pragma: no cover - analytics optional
    from analytics.metrics_store import record_metric
except Exception:  # pragma: no cover - fallback stub
    def record_metric(*a, **k):  # type: ignore
        return None

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - sklearn may be absent in tests
    IsolationForest = None  # type: ignore

try:  # pragma: no cover - alerting optional in tests
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - fallback stub
    def send_alert(msg: str) -> None:  # type: ignore
        return

logger = logging.getLogger(__name__)


@dataclass
class _RunningStat:
    """Welford's streaming variance algorithm."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


@dataclass
class PipelineAnomalyDetector:
    """Stateful detector that tracks anomaly statistics."""

    z_threshold: float = 4.0
    contamination: float = 0.01
    alert_threshold: float = 0.05
    max_history: int = 1024
    feature_stats: Dict[str, _RunningStat] = field(default_factory=dict)
    spread_stat: _RunningStat = field(default_factory=_RunningStat)
    resid_stat: _RunningStat = field(default_factory=_RunningStat)
    _history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=1024))
    _iforest: IsolationForest | None = field(init=False, default=None)
    _iforest_ready: bool = field(init=False, default=False)
    total: int = 0
    anomalies: int = 0

    def __post_init__(self) -> None:
        if IsolationForest is not None:  # pragma: no cover - sklearn optional
            self._iforest = IsolationForest(contamination=self.contamination, random_state=42)

    # ------------------------------------------------------------------
    def _quarantine(self, df: pd.DataFrame) -> None:
        """Persist offending batch for later inspection."""
        try:
            qdir = Path("pipeline_quarantine")
            qdir.mkdir(exist_ok=True)
            fname = qdir / f"batch_{int(time.time()*1000)}.parquet"
            df.to_parquet(fname)
        except Exception as exc:  # pragma: no cover - best effort only
            logger.debug("Failed to quarantine batch: %s", exc)

    # ------------------------------------------------------------------
    def _check_spread(self, df: pd.DataFrame) -> int:
        if not {"Bid", "Ask"}.issubset(df.columns):
            return 0
        spreads = df["Ask"].astype(float) - df["Bid"].astype(float)
        for s in spreads:
            self.spread_stat.update(float(s))
        if self.spread_stat.n > 10 and self.spread_stat.std > 0:
            z = (spreads - self.spread_stat.mean) / self.spread_stat.std
            return int((z.abs() > self.z_threshold).sum())
        return 0

    # ------------------------------------------------------------------
    def _check_features(self, df: pd.DataFrame) -> int:
        num_cols = [
            c
            for c in df.columns
            if c not in {"Timestamp", "Bid", "Ask", "prediction", "target"}
            and np.issubdtype(df[c].dtype, np.number)
        ]
        if not num_cols:
            return 0
        feats = df[num_cols].astype(float)
        count = 0
        for col in num_cols:
            stat = self.feature_stats.setdefault(col, _RunningStat())
            for val in feats[col]:
                stat.update(float(val))
            if stat.n > 10 and stat.std > 0:
                z = (feats[col] - stat.mean) / stat.std
                count += int((z.abs() > self.z_threshold).sum())
        if self._iforest is not None:
            arr = feats.values
            try:  # pragma: no cover - sklearn optional
                if self._iforest_ready:
                    preds = self._iforest.predict(arr)
                    count += int((preds == -1).sum())
                self._history.extend(arr)
                if len(self._history) >= 32:  # train when enough history
                    self._iforest.fit(np.array(self._history))
                    self._iforest_ready = True
            except Exception as exc:  # pragma: no cover
                logger.debug("IsolationForest error: %s", exc)
        return count

    # ------------------------------------------------------------------
    def _check_residuals(self, df: pd.DataFrame) -> int:
        if not {"prediction", "target"}.issubset(df.columns):
            return 0
        resid = df["target"].astype(float) - df["prediction"].astype(float)
        for r in resid:
            self.resid_stat.update(float(r))
        if self.resid_stat.n > 10 and self.resid_stat.std > 0:
            z = (resid - self.resid_stat.mean) / self.resid_stat.std
            return int((z.abs() > self.z_threshold).sum())
        return 0

    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame, *, quarantine: bool = True) -> bool:
        """Return ``True`` if ``df`` passes anomaly checks."""

        if df.empty:
            return True
        bad = 0
        bad += self._check_spread(df)
        bad += self._check_features(df)
        bad += self._check_residuals(df)
        self.total += len(df)
        self.anomalies += bad
        PIPELINE_ANOMALY_TOTAL.inc(bad)
        rate = self.anomalies / self.total if self.total else 0.0
        PIPELINE_ANOMALY_RATE.set(rate)
        record_metric("pipeline_anomaly_total", bad)
        record_metric("pipeline_anomaly_rate", rate)
        if bad:
            logger.warning("Pipeline anomaly detected: %d rows", bad)
            if quarantine:
                self._quarantine(df)
        if rate > self.alert_threshold:
            send_alert(f"Pipeline anomaly rate {rate:.2%} exceeds threshold")
        return bad == 0


def validate(df: pd.DataFrame, *, quarantine: bool = True) -> bool:
    """Validate ``df`` against pipeline anomaly checks."""
    return _DETECTOR.validate(df, quarantine=quarantine)


_DETECTOR = PipelineAnomalyDetector()
