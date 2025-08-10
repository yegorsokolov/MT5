import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0,
    contamination: float = 0.01,
    quarantine_path: Optional[Path] = None,
    counter: Optional["Counter"] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect anomalies in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing numeric columns to inspect.
    method : str, optional
        Detection method: ``"zscore"`` or ``"isolation_forest"``.
    threshold : float, optional
        Z-score threshold when ``method='zscore'``.
    contamination : float, optional
        Expected contamination ratio for IsolationForest.
    quarantine_path : Path, optional
        If provided, anomalous rows are appended to this CSV file.
    counter : Counter, optional
        Prometheus counter incremented with the number of anomalies.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (clean_df, anomalies_df).
    """
    if df.empty:
        return df, df

    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
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

    if quarantine_path is not None and not anomalies.empty:
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        header = not quarantine_path.exists()
        anomalies.to_csv(quarantine_path, mode="a", index=False, header=header)

    if counter is not None and not anomalies.empty:
        counter.inc(len(anomalies))

    return clean, anomalies
