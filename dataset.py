"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import datetime as dt
import functools
from sklearn.decomposition import PCA
from dateutil import parser as date_parser
import requests
import logging

try:
    from plugins import FEATURE_PLUGINS
except Exception:
    FEATURE_PLUGINS = []

NEWS_SOURCES = [
    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
]

logger = logging.getLogger(__name__)


def _get_ff_events() -> List[dict]:
    events = []
    for url in NEWS_SOURCES:
        try:
            logger.debug("Fetching events from %s", url)
            events.extend(requests.get(url, timeout=10).json())
        except Exception as e:
            logger.warning("Failed to fetch events from %s: %s", url, e)
            continue
    logger.debug("Fetched %d events from FF", len(events))
    return events


def _get_tradays_events() -> List[dict]:
    url = "https://www.tradays.com/en/economic-calendar.ics"
    try:
        logger.debug("Fetching events from %s", url)
        text = requests.get(url, timeout=10).text
    except Exception as e:
        logger.warning("Failed to fetch events from %s: %s", url, e)
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
    logger.debug("Fetched %d events from Tradays", len(events))
    return events


def _get_mql5_events() -> List[dict]:
    try:
        import MetaTrader5 as mt5
    except Exception:
        return []
    if not mt5.initialize():
        logger.warning("Failed to initialize MetaTrader5 for events")
        return []
    now = dt.datetime.now(tz=dt.timezone.utc)
    start = now - dt.timedelta(days=1)
    end = now + dt.timedelta(days=7)
    logger.debug("Fetching events from MetaTrader5")
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
    logger.debug("Fetched %d events from MetaTrader5", len(events))
    return events


@functools.lru_cache
def get_events(past_events: bool = False) -> List[dict]:
    """Download economic calendar events from multiple sources."""
    logger.info("Fetching economic events")
    events = []
    events.extend(_get_ff_events())
    events.extend(_get_tradays_events())
    events.extend(_get_mql5_events())

    now = dt.datetime.now(tz=dt.timezone.utc)
    filtered = []
    for e in events:
        try:
            date = e["date"] = (
                date_parser.parse(e["date"])
                if isinstance(e["date"], str)
                else e["date"]
            )
        except Exception:
            continue
        if past_events or date >= now:
            filtered.append(e)
    logger.info("Total events returned: %d", len(filtered))
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
            logger.info("Downloading history from %s", url)
            gdown.download(url, str(dest), quiet=False)
            dfs.append(pd.read_csv(dest))

    df = pd.concat(dfs, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d %H:%M:%S:%f")
    logger.info("Loaded %d rows from URLs", len(df))
    return df


def _find_mt5_symbol(symbol: str):
    """Return the matching MetaTrader 5 symbol name, trying common prefixes and suffixes."""
    import MetaTrader5 as mt5

    info = mt5.symbol_info(symbol)
    if info:
        return symbol

    all_symbols = mt5.symbols_get()
    for s in all_symbols:
        name = s.name
        if name.endswith(symbol) or name.startswith(symbol):
            return name
    return None


def load_history_mt5(symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download tick history from the MetaTrader 5 terminal history center."""
    try:
        import MetaTrader5 as mt5
    except Exception as e:
        raise ImportError("MetaTrader5 package is required") from e

    if not mt5.initialize():
        logger.error("Failed to initialize MetaTrader5")
        raise RuntimeError("Failed to initialize MetaTrader5")

    real_sym = _find_mt5_symbol(symbol)
    if not real_sym:
        mt5.shutdown()
        raise ValueError(f"Symbol {symbol} not found in MetaTrader5")

    mt5.symbol_select(real_sym, True)

    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    chunk = 86400 * 7  # request one week at a time to avoid server limits

    ticks = []
    cur = start_ts
    while cur < end_ts:
        to = min(cur + chunk, end_ts)
        logger.debug("Requesting ticks %s - %s", cur, to)
        arr = mt5.copy_ticks_range(real_sym, cur, to, mt5.COPY_TICKS_ALL)
        if arr is not None and len(arr) > 0:
            ticks.extend(arr)
        cur = to

    mt5.shutdown()

    if not ticks:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)
    df["Timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df.rename(columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True)
    df = df[["Timestamp", "Bid", "Ask", "Volume"]]
    df["BidVolume"] = df["Volume"]
    df["AskVolume"] = df["Volume"]
    df.drop(columns=["Volume"], inplace=True)
    logger.info("Loaded %d ticks from MetaTrader5", len(df))
    return df


def load_history_config(sym: str, cfg: dict, root: Path) -> pd.DataFrame:
    """Load history for ``sym`` using local files, URLs or APIs."""
    csv_path = root / "data" / f"{sym}_history.csv"
    pq_path = root / "data" / f"{sym}_history.parquet"
    if pq_path.exists():
        logger.info("Loading history for %s from %s", sym, pq_path)
        return load_history_parquet(pq_path)
    if csv_path.exists():
        logger.info("Loading history for %s from %s", sym, csv_path)
        return load_history(csv_path)

    api_cfg = (cfg.get("api_history") or {}).get(sym)
    if api_cfg:
        provider = api_cfg.get("provider", "mt5")
        start = pd.to_datetime(api_cfg.get("start"))
        end = pd.to_datetime(api_cfg.get("end"))
        if provider == "mt5":
            logger.info("Downloading history for %s from MetaTrader5", sym)
            df = load_history_mt5(sym, start, end)
        else:
            raise ValueError(f"Unknown history provider {provider}")
        save_history_parquet(df, pq_path)
        return df

    urls = cfg.get("data_urls", {}).get(sym)
    if urls:
        logger.info("Downloading history for %s from URLs", sym)
        df = load_history_from_urls(urls)
        save_history_parquet(df, pq_path)
        return df

    raise FileNotFoundError(f"No history found for {sym} and no data source configured")


def load_history(path: Path) -> pd.DataFrame:
    """Load historical tick data from CSV."""
    logger.info("Loading CSV history from %s", path)
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d %H:%M:%S:%f")
    logger.debug("Loaded %d rows from CSV", len(df))
    return df


def load_history_parquet(path: Path) -> pd.DataFrame:
    """Load historical tick data stored in a Parquet file."""
    logger.info("Loading Parquet history from %s", path)
    df = pd.read_parquet(path)
    if "Timestamp" in df.columns:
        # ensure the Timestamp column is parsed as datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)
    logger.debug("Loaded %d rows from Parquet", len(df))
    return df


def save_history_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save tick history to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True).dt.tz_localize(
            None
        )
    logger.info("Saving history to %s", path)
    data.to_parquet(path, index=False)


def load_multiple_histories(paths: Dict[str, Path]) -> pd.DataFrame:
    """Load and concatenate history files for multiple symbols."""
    dfs = []
    for symbol, p in paths.items():
        logger.info("Loading history for %s from %s", symbol, p)
        if p.suffix == ".parquet":
            df = load_history_parquet(p)
        else:
            df = load_history(p)
        df["Symbol"] = symbol
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d total rows", len(combined))
    return combined


def load_index_ohlc(source: str) -> pd.DataFrame:
    """Load daily OHLC data for an index from a local path or URL."""
    try:
        if str(source).startswith("http"):
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(Path(source))
    except Exception:
        return pd.DataFrame()
    if "Date" not in df.columns:
        df.columns = [c.strip() for c in df.columns]
    return df


def add_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily index returns/volatility features using forward fill."""
    try:
        from utils import load_config

        cfg = load_config()
        index_sources = cfg.get("index_data", {})
    except Exception:
        index_sources = {}

    logger.debug("Adding index features: %s", list(index_sources.keys()))

    df = df.sort_values("Timestamp")

    for name, src in index_sources.items():
        idx_df = load_index_ohlc(src)
        if {"Date", "Close"}.issubset(idx_df.columns):
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            idx_df = idx_df.sort_values("Date")
            idx_df["return"] = idx_df["Close"].pct_change()
            idx_df["vol"] = idx_df["return"].rolling(21).std()
            idx_df = idx_df[["Date", "return", "vol"]]
            idx_df = idx_df.rename(
                columns={
                    "Date": "index_date",
                    "return": f"{name}_ret",
                    "vol": f"{name}_vol",
                }
            )
            df = pd.merge_asof(
                df,
                idx_df,
                left_on="Timestamp",
                right_on="index_date",
                direction="backward",
            ).drop(columns=["index_date"])
        else:
            df[f"{name}_ret"] = 0.0
            df[f"{name}_vol"] = 0.0

    if not index_sources:
        # provide default columns when config not set
        for col in ["sp500_ret", "sp500_vol", "vix_ret", "vix_vol"]:
            if col not in df.columns:
                df[col] = 0.0
    return df


def add_economic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Adding economic calendar features")
    try:
        events = get_events(past_events=True)
    except Exception:
        df["minutes_to_event"] = np.nan
        df["minutes_from_event"] = np.nan
        df["nearest_news_minutes"] = np.nan
        df["upcoming_red_news"] = 0
        return df
    if not events:
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
    logger.debug("Added calendar features for %d rows", len(df))
    return df


def add_news_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge precomputed news sentiment scores or compute using FinBERT/FinGPT."""

    try:
        from utils import load_config

        cfg = load_config()
    except Exception:  # pragma: no cover - config issues shouldn't fail
        cfg = {}

    if cfg.get("use_fingpt_sentiment", False):
        try:
            from plugins.fingpt_sentiment import score_events

            events = get_events(past_events=True)
            news_df = pd.DataFrame(events)
            if not news_df.empty:
                news_df = score_events(news_df)
                news_df = news_df.rename(columns={"date": "Timestamp"})
                news_df["Timestamp"] = pd.to_datetime(news_df["Timestamp"])
                news_df = news_df.sort_values("Timestamp")[["Timestamp", "sentiment"]]
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df = df.sort_values("Timestamp")
                df = pd.merge_asof(df, news_df, on="Timestamp", direction="backward")
                df["news_sentiment"] = df["sentiment"].fillna(0.0)
                df = df.drop(columns=["sentiment"], errors="ignore")
                logger.debug("Added FinGPT sentiment for %d rows", len(df))
                return df
        except Exception as e:  # pragma: no cover - heavy dependency
            logger.warning("Failed to compute FinGPT sentiment: %s", e)

    if cfg.get("use_finbert_sentiment", False):
        try:
            from plugins.finbert_sentiment import score_events

            events = get_events(past_events=True)
            news_df = pd.DataFrame(events)
            if not news_df.empty:
                news_df = score_events(news_df)
                news_df = news_df.rename(columns={"date": "Timestamp"})
                news_df["Timestamp"] = pd.to_datetime(news_df["Timestamp"])
                news_df = news_df.sort_values("Timestamp")[["Timestamp", "sentiment"]]
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df = df.sort_values("Timestamp")
                df = pd.merge_asof(df, news_df, on="Timestamp", direction="backward")
                df["news_sentiment"] = df["sentiment"].fillna(0.0)
                df = df.drop(columns=["sentiment"], errors="ignore")
                logger.debug("Added FinBERT sentiment for %d rows", len(df))
                return df
        except Exception as e:  # pragma: no cover - heavy dependency
            logger.warning("Failed to compute FinBERT sentiment: %s", e)

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
    logger.debug("Added news sentiment for %d rows", len(df))
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical features used by the ML model."""
    logger.info("Creating features for dataframe with %d rows", len(df))

    try:
        from utils import load_config

        cfg = load_config()
    except Exception:
        cfg = {}

    use_atr = cfg.get("use_atr", True)
    use_donchian = cfg.get("use_donchian", True)

    def _feat(group: pd.DataFrame) -> pd.DataFrame:
        mid = (group["Bid"] + group["Ask"]) / 2
        group = group.assign(mid=mid)

        # base return and moving averages
        group["return"] = group["mid"].pct_change()
        group["ma_5"] = group["mid"].rolling(5).mean()
        group["ma_10"] = group["mid"].rolling(10).mean()
        group["ma_30"] = group["mid"].rolling(30).mean()
        group["ma_60"] = group["mid"].rolling(60).mean()

        if use_atr:
            tr = group["mid"].diff().abs()
            group["atr_14"] = tr.rolling(14).mean()
            group["atr_stop_long"] = group["mid"] - group["atr_14"] * 3
            group["atr_stop_short"] = group["mid"] + group["atr_14"] * 3

        if use_donchian:
            period = cfg.get("donchian_period", 20)
            group["donchian_high"] = group["mid"].rolling(period).max()
            group["donchian_low"] = group["mid"].rolling(period).min()
            dc_up = group["mid"] > group["donchian_high"].shift(1)
            dc_down = group["mid"] < group["donchian_low"].shift(1)
            group["donchian_break"] = 0
            group.loc[dc_up, "donchian_break"] = 1
            group.loc[dc_down, "donchian_break"] = -1

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

        # microstructure features
        spread = group["Ask"] - group["Bid"]
        group["spread"] = spread
        group["mid_change"] = group["mid"].diff()
        group["spread_change"] = spread.diff()
        delta_sec = group["Timestamp"].diff().dt.total_seconds().replace(0, np.nan)
        group["trade_rate"] = 1 / delta_sec
        group["quote_revision"] = (
            (group["Bid"].diff() != 0) | (group["Ask"].diff() != 0)
        ).astype(int)

        # order book related features if volumes are present
        if {"BidVolume", "AskVolume"}.issubset(group.columns):
            group["volume_ratio"] = group["BidVolume"] / group["AskVolume"].replace(
                0, pd.NA
            )
            group["volume_imbalance"] = group["BidVolume"] - group["AskVolume"]
            group["depth_imbalance"] = group["volume_imbalance"] / (
                group["BidVolume"] + group["AskVolume"]
            )
            group["depth_imbalance"] = group["depth_imbalance"].replace(
                [np.inf, -np.inf], np.nan
            )
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
        df = df.groupby("Symbol", group_keys=False).apply(_feat)
        pivot = df.pivot_table(index="Timestamp", columns="Symbol", values="return")

        # rolling momentum of each symbol's returns shifted by 1-2 periods
        mom_features = {}
        for sym in pivot.columns:
            for lag in [1, 2]:
                mom = pivot[sym].shift(lag).rolling(10).mean()
                mom_features[f"cross_mom_{sym}_{lag}"] = mom
        mom_df = pd.DataFrame(mom_features, index=pivot.index).reset_index()
        df = df.merge(mom_df, on="Timestamp", how="left")

        if df["Symbol"].nunique() > 1:
            pair_data = {}
            for sym1 in pivot.columns:
                for sym2 in pivot.columns:
                    if sym1 == sym2:
                        continue
                    pair_data[(sym1, sym2)] = pivot[sym1].rolling(30).corr(pivot[sym2])

            pair_df = pd.concat(pair_data, axis=1)
            pair_df.columns = pd.MultiIndex.from_tuples(
                pair_df.columns, names=["Symbol", "Other"]
            )
            pair_df = (
                pair_df.stack(["Symbol", "Other"]).rename("pair_corr").reset_index()
            )
            pair_wide = pair_df.pivot_table(
                index=["Timestamp", "Symbol"], columns="Other", values="pair_corr"
            )
            pair_wide = pair_wide.add_prefix("cross_corr_").reset_index()
            df = df.merge(pair_wide, on=["Timestamp", "Symbol"], how="left")

            lag_corr_df = pd.DataFrame(index=pivot.index)
            for sym1 in pivot.columns:
                for sym2 in pivot.columns:
                    if sym1 == sym2:
                        continue
                    for lag in [3, 4, 5]:
                        col = f"cross_corr_{sym1}_{sym2}_lag{lag}"
                        lag_corr_df[col] = (
                            pivot[sym1].rolling(30).corr(pivot[sym2].shift(lag))
                        )

            if not lag_corr_df.empty:
                lag_corr_df = lag_corr_df.reset_index()
                df = df.merge(lag_corr_df, on="Timestamp", how="left")

        pivot_filled = pivot.fillna(0)
        n_comp = min(3, len(pivot.columns))
        pca = PCA(n_components=n_comp)
        factors = pca.fit_transform(pivot_filled)
        factor_df = pd.DataFrame(
            factors, index=pivot.index, columns=[f"factor_{i+1}" for i in range(n_comp)]
        ).reset_index()
        df = df.merge(factor_df, on="Timestamp", how="left")
    else:
        df = _feat(df)

    df = add_economic_calendar_features(df)
    df = add_news_sentiment_features(df)
    df = add_index_features(df)

    for plugin in FEATURE_PLUGINS:
        try:
            df = plugin(df)
        except Exception:
            pass

    thresh = cfg.get("anomaly_threshold")
    if "anomaly_score" in df.columns:
        if thresh is not None:
            df["skip_trade"] = df["anomaly_score"] > thresh
        else:
            df["skip_trade"] = False

    try:
        from regime import label_regimes

        df = label_regimes(df)
    except Exception:
        df["market_regime"] = 0
    logger.info("Finished feature engineering")
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


def ma_cross_signal(
    df: pd.DataFrame, short: str = "ma_10", long: str = "ma_30"
) -> pd.Series:
    """Return 1 when the short MA crosses above the long MA, -1 on cross below."""
    cross_up = (df[short] > df[long]) & (df[short].shift(1) <= df[long].shift(1))
    cross_down = (df[short] < df[long]) & (df[short].shift(1) >= df[long].shift(1))
    signal = pd.Series(0, index=df.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def train_test_split(
    df: pd.DataFrame, n_train: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def make_sequence_arrays(
    df: pd.DataFrame, features: List[str], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
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
            X_list.append(values[i - seq_len : i])
            y_list.append(targets[i])

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y
