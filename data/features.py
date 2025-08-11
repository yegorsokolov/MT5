"""Feature engineering utilities."""

from __future__ import annotations

import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

from utils.data_backend import get_dataframe_module
from analysis.garch_vol import garch_volatility

pd = get_dataframe_module()

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

from .events import get_events
from .feature_store import FeatureStore
from .versioning import compute_hash
from .macro_features import load_macro_series

try:  # optional plugin system
    from plugins import FEATURE_PLUGINS  # type: ignore
except Exception:  # pragma: no cover - plugins optional
    FEATURE_PLUGINS = []

logger = logging.getLogger(__name__)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose numeric columns should be downcast.

    Returns
    -------
    pd.DataFrame
        DataFrame with numeric columns downcast to more memory efficient types.
    """

    df = df.copy()
    for col in df.select_dtypes(include=["int", "float"]).columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


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
    except Exception:  # pragma: no cover - config issues shouldn't fail
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
        for col in ["sp500_ret", "sp500_vol", "vix_ret", "vix_vol"]:
            if col not in df.columns:
                df[col] = 0.0
    return df


def add_economic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Adding economic calendar features")
    try:
        events = get_events(past_events=True)
    except Exception:  # pragma: no cover - network issues
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

    mode = cfg.get("sentiment_mode", "auto")
    if mode == "auto":
        from utils.resource_monitor import monitor

        monitor.start()
        mode = monitor.capabilities.model_size()
    api_url = cfg.get("sentiment_api_url")

    if cfg.get("use_fingpt_sentiment", False):
        try:
            from plugins.fingpt_sentiment import score_events  # type: ignore

            events = get_events(past_events=True)
            news_df = pd.DataFrame(events)
            if not news_df.empty:
                news_df = score_events(news_df, mode=mode, api_url=api_url)
                news_df = news_df.rename(columns={"date": "Timestamp"})
                news_df["Timestamp"] = pd.to_datetime(news_df["Timestamp"])
                keep_cols = ["Timestamp", "sentiment"]
                if "summary" in news_df.columns:
                    keep_cols.append("summary")
                news_df = news_df.sort_values("Timestamp")[keep_cols]
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df = df.sort_values("Timestamp")
                df = pd.merge_asof(df, news_df, on="Timestamp", direction="backward")
                df["news_sentiment"] = df["sentiment"].fillna(0.0)
                if "summary" in news_df.columns:
                    df["news_summary"] = df["summary"].fillna("")
                df = df.drop(columns=["sentiment", "summary"], errors="ignore")
                logger.debug("Added FinGPT sentiment for %d rows", len(df))
                return df
        except Exception as e:  # pragma: no cover - heavy dependency
            logger.warning("Failed to compute FinGPT sentiment: %s", e)

    if cfg.get("use_finbert_sentiment", False):
        try:
            from plugins.finbert_sentiment import score_events  # type: ignore

            events = get_events(past_events=True)
            news_df = pd.DataFrame(events)
            if not news_df.empty:
                news_df = score_events(news_df, mode=mode, api_url=api_url)
                news_df = news_df.rename(columns={"date": "Timestamp"})
                news_df["Timestamp"] = pd.to_datetime(news_df["Timestamp"])
                news_df = news_df.sort_values("Timestamp")[["Timestamp", "sentiment"]]
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df = df.sort_values("Timestamp")
                df = pd.merge_asof(df, news_df, on="Timestamp", direction="backward")
                df["news_sentiment"] = df["sentiment"].fillna(0.0)
                df["news_summary"] = ""
                df = df.drop(columns=["sentiment"], errors="ignore")
                logger.debug("Added FinBERT sentiment for %d rows", len(df))
                return df
        except Exception as e:  # pragma: no cover - heavy dependency
            logger.warning("Failed to compute FinBERT sentiment: %s", e)

    path = Path(__file__).resolve().parent / "data" / "news_sentiment.csv"
    if not path.exists():
        df["news_sentiment"] = 0.0
        df["news_summary"] = ""
        return df

    news = pd.read_csv(path)
    if "Timestamp" not in news.columns or "sentiment" not in news.columns:
        df["news_sentiment"] = 0.0
        df["news_summary"] = ""
        return df

    news["Timestamp"] = pd.to_datetime(news["Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    news = news.sort_values("Timestamp")
    df = df.sort_values("Timestamp")
    df = pd.merge_asof(df, news, on="Timestamp", direction="backward")
    df["news_sentiment"] = df["sentiment"].fillna(0.0)
    if "summary" in news.columns:
        df["news_summary"] = df["summary"].fillna("")
    else:
        df["news_summary"] = ""
    df = df.drop(columns=["sentiment", "summary"], errors="ignore")
    logger.debug("Added news sentiment for %d rows", len(df))
    return df


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-asset statistics between symbol pairs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``Timestamp``, ``Symbol``, ``mid`` and
        ``return`` columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional cross-asset feature columns. Column
        names encode both symbols, e.g. ``EURUSD_GBPUSD_corr_30``.
    """

    if "Symbol" not in df.columns or df["Symbol"].nunique() < 2:
        return df

    pivot_mid = df.pivot_table(index="Timestamp", columns="Symbol", values="mid")
    pivot_ret = df.pivot_table(index="Timestamp", columns="Symbol", values="return")

    feat: Dict[str, pd.Series] = {}
    symbols = list(pivot_mid.columns)
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i + 1 :]:
            # Rolling correlation of returns
            corr = pivot_ret[s1].rolling(30).corr(pivot_ret[s2])
            feat[f"{s1}_{s2}_corr_30"] = corr

            # Spread between mid prices
            spread = pivot_mid[s1] - pivot_mid[s2]
            feat[f"{s1}_{s2}_spread"] = spread

            # Cointegration p-value (constant across time)
            try:
                from statsmodels.tsa.stattools import coint  # type: ignore

                aligned = pivot_mid[[s1, s2]].dropna()
                if len(aligned) > 30:
                    _, pval, _ = coint(aligned[s1], aligned[s2])
                else:
                    pval = np.nan
            except Exception:  # pragma: no cover - optional dependency
                pval = np.nan
            feat[f"{s1}_{s2}_coint_p"] = pd.Series(pval, index=pivot_mid.index)

    if not feat:
        return df

    feat_df = pd.DataFrame(feat, index=pivot_mid.index).reset_index()
    df = df.merge(feat_df, on="Timestamp", how="left")
    return df


def make_features(df: pd.DataFrame, validate: bool = False) -> pd.DataFrame:
    """Add common technical features used by the ML model.

    Parameters
    ----------
    df : pd.DataFrame
        Input tick dataframe.
    validate : bool, optional
        If True, validate the engineered features against ``FEATURE_SCHEMA``.
    """

    logger.info("Creating features for dataframe with %d rows", len(df))

    try:
        from utils import load_config

        cfg = load_config()
    except Exception:
        cfg = {}

    use_atr = cfg.get("use_atr", True)
    use_donchian = cfg.get("use_donchian", True)
    use_dask = cfg.get("use_dask", False)
    use_cache = cfg.get("use_feature_cache", False)
    dask_url = cfg.get("dask_cluster_url")

    adjacency_matrices: dict | None = None

    cached_store: FeatureStore | None = None
    raw_hash = ""
    if use_cache:
        cached_store = FeatureStore()
        # Hash the raw dataframe via a temporary file using `compute_hash`
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            raw_hash = compute_hash(tmp.name)
        os.unlink(tmp.name)
        symbol_key = (
            "-".join(sorted(df["Symbol"].unique())) if "Symbol" in df.columns else "nosymbol"
        )
        window_key = len(df)
        params_key = {"use_atr": use_atr, "use_donchian": use_donchian}
        cached_df = cached_store.load(symbol_key, window_key, params_key, raw_hash)
        if cached_df is not None:
            logger.info("Loading features from cache %s", cached_store.path)
            return cached_df

    if use_dask:
        import dask.dataframe as dd  # type: ignore
        from dask.distributed import Client  # type: ignore

        try:
            Client(dask_url) if dask_url else Client()
        except Exception:  # pragma: no cover - dask cluster issues
            pass

    def _feat(group: pd.DataFrame) -> pd.DataFrame:
        mid = (group["Bid"] + group["Ask"]) / 2
        group = group.assign(mid=mid)

        # base return and moving averages
        group["return"] = group["mid"].pct_change()
        # GARCH-based volatility estimate
        group["garch_vol"] = garch_volatility(group["return"])
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
        group["ma_h4"] = group["mid"].rolling(240, min_periods=1).mean()
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

        return group

    if "Symbol" in df.columns:
        if use_dask:
            meta = _feat(df.head(2).copy())
            ddf = dd.from_pandas(df, npartitions=cfg.get("dask_partitions", 4))
            ddf = ddf.groupby("Symbol", group_keys=False).apply(lambda x: _feat(x), meta=meta)
            df = ddf.compute()
        else:
            df = df.groupby("Symbol", group_keys=False).apply(_feat)
        pivot = df.pivot_table(index="Timestamp", columns="Symbol", values="return")

        # rolling momentum of each symbol's returns shifted by 1-2 periods
        mom_features: Dict[str, pd.Series] = {}
        for sym in pivot.columns:
            for lag in [1, 2]:
                mom = pivot[sym].shift(lag).rolling(10).mean()
                mom_features[f"cross_mom_{sym}_{lag}"] = mom
        mom_df = pd.DataFrame(mom_features, index=pivot.index).reset_index()
        df = df.merge(mom_df, on="Timestamp", how="left")

        adjacency_matrices = None
        if df["Symbol"].nunique() > 1:
            pair_data: Dict[tuple, pd.Series] = {}
            for sym1 in pivot.columns:
                for sym2 in pivot.columns:
                    if sym1 == sym2:
                        continue
                    pair_data[(sym1, sym2)] = pivot[sym1].rolling(30).corr(pivot[sym2])

            symbols = list(pivot.columns)
            adjacency_matrices = {}
            for ts in pivot.index:
                mat = np.zeros((len(symbols), len(symbols)))
                for i, s1 in enumerate(symbols):
                    for j, s2 in enumerate(symbols):
                        if s1 == s2:
                            continue
                        val = pair_data[(s1, s2)].loc[ts]
                        mat[i, j] = 0.0 if pd.isna(val) else float(val)
                adjacency_matrices[ts] = mat

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

            if cfg.get("use_pair_trading", False):
                from statsmodels.tsa.stattools import coint  # type: ignore
                window = cfg.get("pair_z_window", 20)
                pivot_mid = df.pivot_table(index="Timestamp", columns="Symbol", values="mid")
                pair_feat = pd.DataFrame(index=pivot_mid.index)
                for s1 in pivot_mid.columns:
                    for s2 in pivot_mid.columns:
                        if s1 >= s2:
                            continue
                        pair_name = f"{s1}_{s2}"
                        aligned = pivot_mid[[s1, s2]].dropna()
                        if len(aligned) < window + 5:
                            continue
                        beta, alpha = np.polyfit(aligned[s2], aligned[s1], 1)
                        spread = pivot_mid[s1] - (beta * pivot_mid[s2] + alpha)
                        z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
                        pair_feat[f"pair_z_{pair_name}"] = z
                        try:
                            _, pval, _ = coint(aligned[s1], aligned[s2])
                        except Exception:
                            pval = np.nan
                        pair_feat[f"pair_coint_p_{pair_name}"] = pval

                if not pair_feat.empty:
                    pair_feat = pair_feat.reset_index()
                    df = df.merge(pair_feat, on="Timestamp", how="left")

        pivot_filled = pivot.fillna(0)
        if (
            pivot_filled.empty
            or pivot_filled.shape[1] == 0
            or len(pivot_filled) < 2
        ):
            factor_df = pd.DataFrame(index=pivot.index)
        else:
            n_comp = min(3, len(pivot.columns))
            pca = PCA(n_components=n_comp)
            factors = pca.fit_transform(pivot_filled)
            factor_df = pd.DataFrame(
                factors,
                index=pivot.index,
                columns=[f"factor_{i+1}" for i in range(n_comp)],
            )
        if not factor_df.empty:
            factor_df = factor_df.reset_index()
            df = df.merge(factor_df, on="Timestamp", how="left")
    else:
        if use_dask:
            ddf = dd.from_pandas(df, npartitions=cfg.get("dask_partitions", 4))
            df = ddf.map_partitions(_feat).compute()
        else:
            df = _feat(df)

    timeframes = cfg.get("multi_timeframes", [])
    if timeframes:
        from .multitimeframe import aggregate_timeframes

        agg_df = aggregate_timeframes(df, timeframes)
        df = df.merge(agg_df, on="Timestamp", how="left")

    df = add_economic_calendar_features(df)
    df = add_news_sentiment_features(df)
    df = add_index_features(df)
    df = add_cross_asset_features(df)

    macro_symbols = cfg.get("macro_series", [])
    if macro_symbols:
        macro_df = load_macro_series(macro_symbols)
        if not macro_df.empty:
            macro_df = macro_df.sort_values("Date").rename(columns={"Date": "macro_date"})
            df = df.sort_values("Timestamp")
            df = pd.merge_asof(
                df,
                macro_df,
                left_on="Timestamp",
                right_on="macro_date",
                direction="backward",
            )
            df = df.drop(columns=["macro_date"])
            rename_map = {sym: f"macro_{sym}" for sym in macro_symbols if sym in macro_df.columns}
            df = df.rename(columns=rename_map)
        for sym in macro_symbols:
            col = f"macro_{sym}"
            if col not in df.columns:
                df[col] = np.nan

    if adjacency_matrices is None:
        if "Symbol" in df.columns:
            syms = sorted(df["Symbol"].unique())
            adjacency_matrices = {
                ts: np.zeros((len(syms), len(syms)))
                for ts in df["Timestamp"].unique()
            }
        else:
            adjacency_matrices = {}

    for plugin in FEATURE_PLUGINS:
        try:
            if "adjacency_matrices" in inspect.signature(plugin).parameters:
                df = plugin(df, adjacency_matrices=adjacency_matrices)
            else:
                df = plugin(df)
        except Exception:  # pragma: no cover - plugin failure
            pass

    thresh = cfg.get("anomaly_threshold")
    if "anomaly_score" in df.columns:
        if thresh is not None:
            df["skip_trade"] = df["anomaly_score"] > thresh
        else:
            df["skip_trade"] = False

    try:
        from regime import label_regimes  # type: ignore

        df = label_regimes(df)
    except Exception:  # pragma: no cover - optional dependency
        df["market_regime"] = 0
    logger.info("Finished feature engineering")

    df = optimize_dtypes(df)
    if validate:
        from .validators import FEATURE_SCHEMA

        FEATURE_SCHEMA.validate(df, lazy=True)
    if use_cache and cached_store is not None:
        symbol_key = (
            "-".join(sorted(df["Symbol"].unique())) if "Symbol" in df.columns else "nosymbol"
        )
        window_key = len(df)
        params_key = {"use_atr": use_atr, "use_donchian": use_donchian}
        cached_store.save(df, symbol_key, window_key, params_key, raw_hash)
        logger.info("Cached features written to %s", cached_store.path)

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
        trains: List[pd.DataFrame] = []
        tests: List[pd.DataFrame] = []
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
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

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

__all__ = [
    "add_index_features",
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "add_cross_asset_features",
    "make_features",
    "compute_rsi",
    "ma_cross_signal",
    "train_test_split",
    "make_sequence_arrays",
]
