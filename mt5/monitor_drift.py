from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import mt5.log_utils as log_utils

import pandas as pd
import numpy as np

from analytics.metrics_store import record_metric
from analysis.feature_selector import (
    load_feature_set,
    save_feature_set,
    select_features,
)
try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests
    def send_alert(msg: str) -> None:  # type: ignore
        return

LOG_DIR = getattr(log_utils, "LOG_DIR", Path(__file__).resolve().parents[1] / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_METRICS = LOG_DIR / "drift_metrics.parquet"
BASELINE_METRICS = LOG_DIR / "training_baseline.parquet"


def _pyarrow_engine() -> Optional[str]:
    """Return the configured parquet engine when ``pyarrow`` is available."""

    try:
        import pyarrow  # type: ignore  # noqa: F401
    except ImportError:
        return None
    return "pyarrow"


def _write_parquet_store(df: pd.DataFrame, path: Path) -> None:
    """Persist ``df`` to ``path`` while safely appending existing batches."""

    path.parent.mkdir(parents=True, exist_ok=True)
    engine = _pyarrow_engine()
    read_kwargs: dict[str, str] = {}
    write_kwargs: dict[str, str] = {}
    if engine is not None:
        read_kwargs["engine"] = engine
        write_kwargs["engine"] = engine
    if path.exists():
        try:
            existing = pd.read_parquet(path, **read_kwargs)
        except (ImportError, ValueError):
            existing = pd.read_parquet(path)
        df = pd.concat([existing, df], sort=False)
    try:
        df.to_parquet(path, **write_kwargs)
    except (ImportError, ValueError):
        df.to_parquet(path)


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Compute the Population Stability Index between two samples.

    Parameters
    ----------
    expected, actual:
        Series representing baseline and current distributions.
    buckets:
        Number of quantile-based buckets to use.
    """

    quantiles = expected.quantile(np.linspace(0, 1, buckets + 1))
    breaks = np.unique(quantiles.values)
    if len(breaks) == 1:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=breaks)
    a_counts, _ = np.histogram(actual, bins=breaks)
    if a_counts.sum() == 0 or e_counts.sum() == 0:
        return float("inf")
    e_perc = e_counts / e_counts.sum()
    a_perc = a_counts / a_counts.sum()
    e_perc = np.where(e_perc == 0, 1e-6, e_perc)
    a_perc = np.where(a_perc == 0, 1e-6, a_perc)
    return float(((e_perc - a_perc) * np.log(e_perc / a_perc)).sum())


def ks_2samp(sample1: pd.Series, sample2: pd.Series) -> tuple[float, float]:
    """Simple two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    sample1, sample2:
        Samples to compare.

    Returns
    -------
    statistic, pvalue
    """

    data1 = np.sort(sample1.dropna())
    data2 = np.sort(sample2.dropna())
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side="right") / len(data1)
    cdf2 = np.searchsorted(data2, data_all, side="right") / len(data2)
    d = np.max(np.abs(cdf1 - cdf2))
    n1, n2 = len(data1), len(data2)
    en = np.sqrt(n1 * n2 / (n1 + n2))
    pvalue = 2 * np.exp(-2 * (en * d) ** 2)
    return float(d), float(pvalue)


class DriftMonitor:
    """Capture feature/prediction distributions and detect drift."""

    def __init__(
        self,
        baseline_path: Path = BASELINE_METRICS,
        store_path: Path = DRIFT_METRICS,
        threshold: float = 0.05,
        drift_threshold: float = 0.2,
        feature_set_path: Path = (LOG_DIR / "nn_artifacts" / "selected_features.json"),
    ) -> None:
        self.baseline_path = baseline_path
        self.store_path = store_path
        self.threshold = threshold
        self.drift_threshold = drift_threshold
        self.feature_set_path = feature_set_path
        self.feature_set_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._task: Optional[asyncio.Task[None]] = None

    def record(self, features: pd.DataFrame, preds: pd.Series) -> None:
        """Append feature and prediction samples to the drift store."""
        df = features.copy()
        df["prediction"] = preds
        _write_parquet_store(df, self.store_path)

    def compare(self) -> None:
        """Compare current distributions against training baselines."""
        if not (self.baseline_path.exists() and self.store_path.exists()):
            return
        baseline = pd.read_parquet(self.baseline_path)
        current = pd.read_parquet(self.store_path)
        drift_triggered = False
        for col in baseline.columns:
            if col not in current.columns:
                continue
            stat, pval = ks_2samp(baseline[col].dropna(), current[col].dropna())
            if pval < self.threshold:
                self.logger.warning(
                    "Data drift detected for %s (p=%.4f)", col, pval
                )
                try:
                    record_metric("drift_events", 1)
                except Exception:
                    pass
                send_alert(
                    f"Data drift detected for {col} (p={pval:.4f})"
                )
        psi_scores: dict[str, float] = {}
        for col in baseline.columns:
            if col in {"prediction", "target"} or col not in current.columns:
                continue
            psi_scores[col] = population_stability_index(
                baseline[col].dropna(), current[col].dropna()
            )
        for col, score in psi_scores.items():
            if score > self.drift_threshold:
                self.logger.warning(
                    "Feature drift detected for %s (psi=%.4f)", col, score
                )
                drift_triggered = True
        if drift_triggered and "target" in current.columns:
            feats = current.drop(columns=[c for c in ["prediction", "target"] if c in current.columns])
            selected = select_features(feats, current["target"])
            save_feature_set(selected, self.feature_set_path)

    async def _periodic_check(self) -> None:
        while True:
            await asyncio.sleep(24 * 60 * 60)
            self.compare()

    def start(self) -> asyncio.Task[None]:
        """Start daily drift monitoring."""

        if self._task is not None:
            if not self._task.done():
                return self._task
            self._task = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "DriftMonitor.start() requires an active running event loop"
            ) from exc
        self._task = loop.create_task(self._periodic_check())
        return self._task

    async def stop(self) -> None:
        """Cancel the background drift monitoring task if running."""

        task = self._task
        if task is None:
            return
        self._task = None
        if task.done():
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                self.logger.exception("Drift monitor task terminated with error")
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except RuntimeError as exc:
            if "Event loop is closed" not in str(exc):
                raise
        except Exception:
            self.logger.exception("Drift monitor task terminated with error")


monitor = DriftMonitor()


def get_monitor() -> DriftMonitor:
    """Return the module-level drift monitor instance."""

    return monitor


def start_monitoring(drift_monitor: DriftMonitor | None = None) -> asyncio.Task[None]:
    """Start periodic monitoring for ``drift_monitor`` or the default instance."""

    monitor_obj = drift_monitor or monitor
    return monitor_obj.start()


async def stop_monitoring(drift_monitor: DriftMonitor | None = None) -> None:
    """Stop periodic monitoring for ``drift_monitor`` or the default instance."""

    monitor_obj = drift_monitor or monitor
    await monitor_obj.stop()
