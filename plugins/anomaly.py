from . import register_feature
from utils import load_config
from sklearn.ensemble import IsolationForest
import pandas as pd

@register_feature
def add_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add an IsolationForest anomaly score computed on returns."""
    cfg = load_config()
    if not cfg.get("use_anomaly_check", False):
        df["anomaly_score"] = 0.0
        return df

    if "return" not in df.columns:
        return df

    def _score(series: pd.Series) -> pd.Series:
        arr = series.fillna(0).values.reshape(-1, 1)
        if len(arr) < 10:
            return pd.Series(0.0, index=series.index)
        model = IsolationForest(random_state=42, contamination=0.05)
        model.fit(arr)
        return pd.Series(-model.decision_function(arr), index=series.index)

    if "Symbol" in df.columns:
        df["anomaly_score"] = df.groupby("Symbol", group_keys=False)["return"].apply(_score)
    else:
        df["anomaly_score"] = _score(df["return"])

    return df
