"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import datetime as dt
import functools
from dateutil import parser as date_parser
import requests

NEWS_SOURCES = [
    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
]


def _get_ff_events() -> List[dict]:
    events = []
    for url in NEWS_SOURCES:
        try:
            events.extend(requests.get(url, timeout=10).json())
        except Exception:
            continue
    return events


def _get_tradays_events() -> List[dict]:
    url = "https://www.tradays.com/en/economic-calendar.ics"
    try:
        text = requests.get(url, timeout=10).text
    except Exception:
        return []
    events = []
    cur: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line == "BEGIN:VEVENT":
            cur = {}
        elif line == "END:VEVENT":
            if "DTSTART" in cur:
                events.append(
                    {
                        "date": cur["DTSTART"],
                        "impact": cur.get("IMPORTANCE", "Medium"),
                        "currency": cur.get("CURRENCY", ""),
                        "event": cur.get("SUMMARY", ""),
                    }
                )
        elif ":" in line:
            key, val = line.split(":", 1)
            cur[key] = val
    return events


def _get_mql5_events() -> List[dict]:
    try:
        import MetaTrader5 as mt5
    except Exception:
        return []
    if not mt5.initialize():
        return []
    now = dt.datetime.now(tz=dt.timezone.utc)
    start = now - dt.timedelta(days=1)
    end = now + dt.timedelta(days=7)
    values = mt5.calendar_value_history(from_date=start, to_date=end)
    mt5.shutdown()
    if values is None:
        return []
    events = []
    for v in values:
        try:
            event_time = dt.datetime.fromtimestamp(v.time, tz=dt.timezone.utc)
        except Exception:
            continue
        impact = getattr(v, "importance", 1)
        impact_map = {0: "Low", 1: "Medium", 2: "High"}
        events.append(
            {
                "date": event_time.isoformat(),
                "impact": impact_map.get(impact, "Medium"),
                "currency": getattr(v, "currency", ""),
                "event": getattr(v, "event", ""),
            }
        )
    return events


@functools.lru_cache
def get_events(past_events: bool = False) -> List[dict]:
    """Download economic calendar events from multiple sources."""
    events = []
    events.extend(_get_ff_events())
    events.extend(_get_tradays_events())
    events.extend(_get_mql5_events())

    now = dt.datetime.now(tz=dt.timezone.utc)
    filtered = []
    for e in events:
        try:
            date = e["date"] = (
                date_parser.parse(e["date"]) if isinstance(e["date"], str) else e["date"]
            )
        except Exception:
            continue
        if past_events or date >= now:
            filtered.append(e)
    return filtered



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




def add_economic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        events = get_events(past_events=True)
    except Exception:
        df["minutes_to_event"] = np.nan
        df["minutes_from_event"] = np.nan
        df["nearest_news_minutes"] = np.nan
        df["upcoming_red_news"] = 0
        return df
    events_df = pd.DataFrame(events)
    events_df = events_df[events_df["impact"].isin(["High", "Medium"])]
    if events_df.empty:
        df["minutes_to_event"] = np.nan
        df["minutes_from_event"] = np.nan
        df["nearest_news_minutes"] = np.nan
        df["upcoming_red_news"] = 0
        return df
    events_df["date"] = pd.to_datetime(events_df["date"], utc=True)
    events_df = events_df.sort_values("date")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values("Timestamp")

    fwd = pd.merge_asof(
        df[["Timestamp"]],
        events_df[["date"]].rename(columns={"date": "event_time"}),
        left_on="Timestamp",
        right_on="event_time",
        direction="forward",
    )
    bwd = pd.merge_asof(
        df[["Timestamp"]],
        events_df[["date"]].rename(columns={"date": "event_time"}),
        left_on="Timestamp",
        right_on="event_time",
        direction="backward",
    )

    df["minutes_to_event"] = (
        fwd["event_time"] - df["Timestamp"]
    ).dt.total_seconds() / 60
    df["minutes_from_event"] = (
        df["Timestamp"] - bwd["event_time"]
    ).dt.total_seconds() / 60

    nearest = np.vstack(
        [
            df["minutes_to_event"].abs().fillna(np.inf),
            df["minutes_from_event"].abs().fillna(np.inf),
        ]
    ).min(axis=0)
    df["nearest_news_minutes"] = nearest
    df["upcoming_red_news"] = (
        (df["minutes_to_event"] >= 0) & (df["minutes_to_event"] <= 60)
    ).astype(int)
    return df


def add_news_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge precomputed news sentiment scores if available."""
    path = Path(__file__).resolve().parent / "data" / "news_sentiment.csv"
    if not path.exists():
        df["news_sentiment"] = 0.0
        return df

    news = pd.read_csv(path)
    if "Timestamp" not in news.columns or "sentiment" not in news.columns:
        df["news_sentiment"] = 0.0
        return df

    news["Timestamp"] = pd.to_datetime(news["Timestamp"])  # ensure datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    news = news.sort_values("Timestamp")
    df = df.sort_values("Timestamp")
    df = pd.merge_asof(df, news, on="Timestamp", direction="backward")
    df["news_sentiment"] = df["sentiment"].fillna(0.0)
    df = df.drop(columns=["sentiment"], errors="ignore")
    return df
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

        # Bollinger bands (20 period) and breakout signal
        group["ma_h4"] = group["mid"].rolling(240).mean()
        boll_ma = group["mid"].rolling(20).mean()
        boll_std = group["mid"].rolling(20).std()
        group["boll_upper"] = boll_ma + 2 * boll_std
        group["boll_lower"] = boll_ma - 2 * boll_std
        break_up = (group["mid"] > group["boll_upper"]) & (
            group["mid"].shift(1) <= group["boll_upper"].shift(1)
        )
        break_down = (group["mid"] < group["boll_lower"]) & (
            group["mid"].shift(1) >= group["boll_lower"].shift(1)
        )
        group["boll_break"] = 0
        group.loc[break_up, "boll_break"] = 1
        group.loc[break_down, "boll_break"] = -1

        # volatility measure and RSI
        group["volatility_30"] = group["return"].rolling(30).std()
        group["rsi_14"] = compute_rsi(group["mid"], 14)

        # order book related features if volumes are present
        if {"BidVolume", "AskVolume"}.issubset(group.columns):
            spread = group["Ask"] - group["Bid"]
            group["spread"] = spread
            group["volume_ratio"] = group["BidVolume"] / group["AskVolume"].replace(0, pd.NA)
            group["volume_imbalance"] = group["BidVolume"] - group["AskVolume"]
            total_volume = group["BidVolume"] + group["AskVolume"]
            avg_volume = total_volume.rolling(20).mean()
            group["volume_spike"] = (total_volume > avg_volume * 1.5).astype(int)
        group["hour"] = group["Timestamp"].dt.hour + group["Timestamp"].dt.minute / 60
        group["hour_sin"] = np.sin(2 * np.pi * group["hour"] / 24)
        group["hour_cos"] = np.cos(2 * np.pi * group["hour"] / 24)

        group = group.dropna().reset_index(drop=True)
        group["ma_cross"] = ma_cross_signal(group)
        return group

    if "Symbol" in df.columns:
        df = df.groupby('Symbol', group_keys=False).apply(_feat)
        if df['Symbol'].nunique() > 1:
            pivot = df.pivot_table(index='Timestamp', columns='Symbol', values='return')
            corr_features = {}
            mom_features = {}
            for sym in pivot.columns:
                others = pivot.drop(columns=sym).mean(axis=1)
                corr_features[sym] = pivot[sym].rolling(30).corr(others)
                mom_features[sym] = others.rolling(30).mean()
            corr_df = (
                pd.DataFrame(corr_features)
                .stack()
                .rename('cross_corr')
                .reset_index()
                .rename(columns={'level_0':'Timestamp','level_1':'Symbol'})
            )
            mom_df = (
                pd.DataFrame(mom_features)
                .stack()
                .rename('cross_momentum')
                .reset_index()
                .rename(columns={'level_0':'Timestamp','level_1':'Symbol'})
            )
            df = df.merge(corr_df, on=['Timestamp','Symbol'], how='left')
            df = df.merge(mom_df, on=['Timestamp','Symbol'], how='left')
        else:
            df['cross_corr'] = np.nan
            df['cross_momentum'] = np.nan
    else:
        df = _feat(df)
        df['cross_corr'] = np.nan
        df['cross_momentum'] = np.nan

    df = add_economic_calendar_features(df)
    df = add_news_sentiment_features(df)
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


def make_sequence_arrays(df: pd.DataFrame, features: List[str], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a dataframe to sequences suitable for neural network models."""
    X_list = []
    y_list = []

    groups = [df]
    if "Symbol" in df.columns:
        groups = [g for _, g in df.groupby("Symbol")]

    for g in groups:
        values = g[features].values
        targets = (g["return"].shift(-1) > 0).astype(int).values
        for i in range(seq_len, len(g) - 1):
            X_list.append(values[i - seq_len:i])
            y_list.append(targets[i])

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y
