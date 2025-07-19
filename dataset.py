"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple, List, Dict
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


def load_multiple_histories(paths: Dict[str, Path]) -> pd.DataFrame:
    """Load and concatenate history files for multiple symbols."""
    dfs = []
    for symbol, p in paths.items():
        df = load_history(p)
        df["Symbol"] = symbol
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical features used by the ML model."""

    def _feat(group: pd.DataFrame) -> pd.DataFrame:
        mid = (group["Bid"] + group["Ask"]) / 2
        group = group.assign(mid=mid)

        # base return and moving averages
        group["return"] = group["mid"].pct_change()
        group["ma_5"] = group["mid"].rolling(5).mean()
        group["ma_10"] = group["mid"].rolling(10).mean()
        group["ma_30"] = group["mid"].rolling(30).mean()
        group["ma_60"] = group["mid"].rolling(60).mean()

        # volatility measure and RSI
        group["volatility_30"] = group["return"].rolling(30).std()
        group["rsi_14"] = compute_rsi(group["mid"], 14)

        # order book related features if volumes are present
        if {"BidVolume", "AskVolume"}.issubset(group.columns):
            spread = group["Ask"] - group["Bid"]
            group["spread"] = spread
            group["volume_ratio"] = group["BidVolume"] / group["AskVolume"].replace(0, pd.NA)
            group["volume_imbalance"] = group["BidVolume"] - group["AskVolume"]

        group = group.dropna().reset_index(drop=True)
        group["ma_cross"] = ma_cross_signal(group)
        return group

    if "Symbol" in df.columns:
        df = df.groupby("Symbol", group_keys=False).apply(_feat)
    else:
        df = _feat(df)
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
    """Simple ordered train/test split. When multiple symbols are present
    the split is applied per symbol."""
    if "Symbol" in df.columns:
        trains = []
        tests = []
        for _, group in df.groupby("Symbol"):
            trains.append(group.iloc[:n_train].copy())
            tests.append(group.iloc[n_train:].copy())
        train = pd.concat(trains, ignore_index=True)
        test = pd.concat(tests, ignore_index=True)
        return train, test
    else:
        train = df.iloc[:n_train].copy()
        test = df.iloc[n_train:].copy()
        return train, test
