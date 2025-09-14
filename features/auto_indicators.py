from __future__ import annotations

"""Hypernetwork driven indicator discovery and persistence.

The implementation is dependency light; if pandas is unavailable the functions
operate on simple ``dict`` inputs mapping column name to a list of values.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Sequence

try:  # optional heavy deps
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas may be absent
    pd = None

# Persist auto-discovered indicator descriptors in the feature store so that
# subsequent training runs can automatically enrich the feature matrix.
REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "feature_store" / "auto_indicators.json"
)

logger = logging.getLogger(__name__)


def _load_registry(path: Path = REGISTRY_PATH) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = path.read_text()
        if text.strip():
            return json.loads(text)
    except Exception:
        pass
    return []


def _save_registry(entries: Iterable[Dict[str, Any]], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(list(entries), f, indent=2)


def _basic_compute_dict(series: Sequence[float], lag: int, window: int) -> Dict[str, Sequence[float]]:
    lagged = [None] * lag + list(series[:-lag]) if len(series) else []
    means: list[float] = []
    for i in range(len(series)):
        win = series[max(0, i - window + 1) : i + 1]
        means.append(sum(win) / len(win))
    return {"lag": lagged, "mean": means}


def generate(
    df: Any,
    model: Any,
    asset_features: Sequence[float],
    regime: Sequence[float],
    registry_path: Path = REGISTRY_PATH,
) -> Tuple[Any, Dict[str, int]]:
    """Return structure with generated indicator columns and descriptor."""

    x = list(asset_features) + list(regime)
    lag, window = model(x)
    desc: Dict[str, int] = {"lag": lag, "window": window}

    if pd is not None and isinstance(df, pd.DataFrame):
        out = df.copy()
        col = df.columns[0]
        ser = df[col]
        out[f"{col}_lag{lag}"] = [None] * lag + list(ser.values[:-lag])
        out[f"{col}_mean{window}"] = ser.rolling(window).mean()
        return out, desc
    else:
        data = {k: list(v) for k, v in df.items()}
        key = next(iter(data))
        computed = _basic_compute_dict(data[key], lag, window)
        data[f"{key}_lag{lag}"] = computed["lag"]
        data[f"{key}_mean{window}"] = computed["mean"]
        return data, desc


def apply(
    df: Any,
    registry_path: Path = REGISTRY_PATH,
    formula_dir: Path | None = None,
) -> Any:
    """Append all persisted indicator columns to ``df``.

    Parameters
    ----------
    df: Any
        Either a :class:`pandas.DataFrame` or a ``dict`` mapping column name to
        sequences.  The structure mirrors the accepted inputs for
        :func:`generate`.
    registry_path: Path, optional
        Location of the persisted indicator registry.
    formula_dir: Path, optional
        Directory containing evolved indicator formula files.  All files
        matching ``evolved_indicators_v*.json`` will be applied in order.
    """

    entries = _load_registry(registry_path)

    # Support both DataFrame and dictionary inputs similar to ``generate``.
    if pd is not None and isinstance(df, pd.DataFrame):
        out = df.copy()
        col = df.columns[0]
        ser = df[col]
        for item in entries:
            lag = int(item.get("lag", 1))
            window = int(item.get("window", 1))
            out[f"{col}_lag{lag}"] = [None] * lag + list(ser.values[:-lag])
            out[f"{col}_mean{window}"] = ser.rolling(window).mean()

        formulas_path = formula_dir or registry_path.parent
        from . import evolved_indicators

        for file in sorted(formulas_path.glob("evolved_indicators_v*.json")):
            out = evolved_indicators.compute(out, path=file)
        return out
    else:
        data = {k: list(v) for k, v in df.items()}
        key = next(iter(data))
        for item in entries:
            lag = int(item.get("lag", 1))
            window = int(item.get("window", 1))
            computed = _basic_compute_dict(data[key], lag, window)
            data[f"{key}_lag{lag}"] = computed["lag"]
            data[f"{key}_mean{window}"] = computed["mean"]
        # Dict inputs do not support arbitrary formula evaluation
        return data


def persist(
    indicator: Dict[str, int],
    metrics: Dict[str, Any],
    registry_path: Path = REGISTRY_PATH,
) -> None:
    """Persist descriptor and log its improvement metrics."""

    entry: Dict[str, Any] = {**indicator, "metrics": metrics}
    entries = _load_registry(registry_path)
    entries.append(entry)
    _save_registry(entries, registry_path)
    logger.info("Persisted indicator lag=%s window=%s with metrics=%s", indicator.get("lag"), indicator.get("window"), metrics)


__all__ = ["generate", "persist", "apply", "REGISTRY_PATH"]
