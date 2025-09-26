"""Post-processing helpers for the training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, TYPE_CHECKING

import joblib
import pandas as pd

if TYPE_CHECKING:
    from analysis.prob_calibration import ConformalIntervalParams
else:  # pragma: no cover - avoid heavy optional dependency at import time
    ConformalIntervalParams = object  # type: ignore[assignment]

from models import model_store

__all__ = [
    "summarise_predictions",
    "build_model_metadata",
    "persist_model",
]


def summarise_predictions(
    true_labels: Iterable[int],
    preds: Iterable[int],
    probs: Iterable[float],
    regimes: Iterable[int],
    lower: Iterable[float] | None = None,
    upper: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return a dataframe aggregating validation predictions."""

    data = {
        "y_true": list(true_labels),
        "pred": list(preds),
        "prob": list(probs),
        "market_regime": list(regimes),
    }
    if lower is not None:
        data["lower"] = list(lower)
    if upper is not None:
        data["upper"] = list(upper)
    return pd.DataFrame(data)


def build_model_metadata(
    regime_thresholds: Mapping[int | str, float],
    *,
    interval_params: ConformalIntervalParams | None = None,
    interval_quantiles: Mapping[int, float] | None = None,
    interval_alpha: float | None = None,
    interval_coverage: float | None = None,
    interval_coverage_by_regime: Mapping[int, float] | None = None,
    meta_model_id: str | None = None,
) -> dict[str, object]:
    """Assemble the metadata dictionary persisted alongside the model."""

    metadata: dict[str, object] = {
        "regime_thresholds": {int(k): float(v) for k, v in regime_thresholds.items()},
    }
    if interval_params is not None:
        metadata["interval_params"] = interval_params.to_dict()
        metadata["interval_alpha"] = float(interval_params.alpha)
        if interval_params.coverage is not None:
            metadata["interval_coverage"] = float(interval_params.coverage)
        if interval_params.coverage_by_regime:
            metadata["interval_coverage_by_regime"] = {
                int(k): float(v) for k, v in interval_params.coverage_by_regime.items()
            }
    if interval_quantiles:
        metadata["interval_quantiles"] = {int(k): float(v) for k, v in interval_quantiles.items()}
    if interval_alpha is not None:
        metadata.setdefault("interval_alpha", float(interval_alpha))
    if interval_coverage is not None:
        metadata.setdefault("interval_coverage", float(interval_coverage))
    if interval_coverage_by_regime:
        metadata.setdefault(
            "interval_coverage_by_regime",
            {int(k): float(v) for k, v in interval_coverage_by_regime.items()},
        )
    if meta_model_id is not None:
        metadata["meta_model_id"] = meta_model_id
    return metadata


def persist_model(
    model,
    cfg,
    performance: Mapping[str, object],
    *,
    features: Iterable[str],
    root: Path,
    artifacts: Mapping[str, str] | None = None,
) -> str:
    """Persist the trained model and register it in the model store."""

    joblib.dump(model, root / "model.joblib")
    return model_store.save_model(
        model,
        cfg,
        performance,
        features=list(features),
        artifacts=dict(artifacts or {}),
    )
