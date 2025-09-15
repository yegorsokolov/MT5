from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - sklearn optional
    IsolationForest = None  # type: ignore


__all__ = ["detect_anomalies"]


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "isolation_forest",
    threshold: float = 3.0,
    contamination: float = 0.01,
    quarantine_path: Optional[Path] = None,
    counter: Optional["Counter"] = None,
    return_mask: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Detect anomalous rows in ``df``.

    Parameters
    ----------
    df:
        Input dataframe containing numeric columns to inspect.
    method:
        Detection method: ``"zscore"`` or ``"isolation_forest"``.
    threshold:
        Z-score threshold when ``method='zscore'``.
    contamination:
        Expected contamination ratio for IsolationForest.
    quarantine_path:
        If provided, anomalous rows are appended to this CSV file.
    counter:
        Prometheus counter incremented with the number of anomalies.
    return_mask:
        When ``True``, also return a boolean mask where ``True`` indicates a
        non-anomalous row. This is useful for down-weighting rather than
        dropping flagged samples.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        Cleaned dataframe, anomalies dataframe and optionally the boolean mask.
    """
    if df.empty:
        if return_mask:
            return df, df, pd.Series(dtype=bool)
        return df, df

    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        if return_mask:
            return df, df.iloc[0:0], pd.Series(dtype=bool)
        return df, df.iloc[0:0]

    numeric = numeric.fillna(0)

    if method == "zscore":
        mean = numeric.mean()
        std = numeric.std(ddof=0).replace(0, np.nan)
        zscores = ((numeric - mean) / std).abs().fillna(0)
        mask = (zscores > threshold).any(axis=1)
    elif method == "isolation_forest":
        clf = IsolationForest(contamination=contamination, random_state=42)
        pred = clf.fit_predict(numeric)
        mask = pred == -1
    else:
        raise ValueError(f"Unknown method: {method}")

    anomalies = df[mask].copy()
    clean = df[~mask].copy()
    keep_mask = ~mask

    if quarantine_path is not None and not anomalies.empty:
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        header = not quarantine_path.exists()
        anomalies.to_csv(quarantine_path, mode="a", index=False, header=header)

    if counter is not None and not anomalies.empty:
        counter.inc(len(anomalies))

    if return_mask:
        return clean, anomalies, keep_mask
    return clean, anomalies
