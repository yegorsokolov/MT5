"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple, List
import pandas as pd


def load_history_from_urls(urls: List[str]) -> pd.DataFrame:
    """Download multiple CSV files and return a combined DataFrame."""
    import gdown
    from tempfile import TemporaryDirectory

    dfs = []
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, url in enumerate(urls):
            dest = tmp_dir / f"part_{i}.csv"
            gdown.download(url, str(dest), quiet=False)
            dfs.append(pd.read_csv(dest))

    df = pd.concat(dfs, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d %H:%M:%S:%f")
    return df


def load_history(path: Path) -> pd.DataFrame:
    """Load historical tick data from CSV."""
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d %H:%M:%S:%f")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical features used by the ML model."""
    mid = (df["Bid"] + df["Ask"]) / 2
    df = df.assign(mid=mid)
    df["return"] = df["mid"].pct_change()
    df["ma_10"] = df["mid"].rolling(10).mean()
    df["ma_30"] = df["mid"].rolling(30).mean()
    df["rsi_14"] = compute_rsi(df["mid"], 14)
    df = df.dropna().reset_index(drop=True)
    df["ma_cross"] = ma_cross_signal(df)
    return df


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index calculation."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def ma_cross_signal(df: pd.DataFrame, short: str = "ma_10", long: str = "ma_30") -> pd.Series:
    """Return 1 when the short MA crosses above the long MA, -1 on cross below."""
    cross_up = (df[short] > df[long]) & (df[short].shift(1) <= df[long].shift(1))
    cross_down = (df[short] < df[long]) & (df[short].shift(1) >= df[long].shift(1))
    signal = pd.Series(0, index=df.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def train_test_split(df: pd.DataFrame, n_train: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple ordered train/test split."""
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test
