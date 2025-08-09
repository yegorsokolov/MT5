"""Historical market data loading utilities."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .versioning import compute_hash

logger = logging.getLogger(__name__)
DATA_VERSIONS_LOG = Path("logs/data_versions.json")


def _guess_symbol(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_history"):
        return stem[:-8]
    return stem


def _record_data_version(path: Path, symbol: str | None = None) -> None:
    try:
        sym = symbol or _guess_symbol(path)
        digest = compute_hash(path)
        logger.info("Hash for %s: %s", sym, digest)
        DATA_VERSIONS_LOG.parent.mkdir(exist_ok=True)
        if DATA_VERSIONS_LOG.exists():
            versions = json.loads(DATA_VERSIONS_LOG.read_text())
        else:
            versions = {}
        versions[sym] = digest
        DATA_VERSIONS_LOG.write_text(json.dumps(versions, indent=2, sort_keys=True))
    except Exception as e:
        logger.warning("Failed to record data version for %s: %s", path, e)


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
    import MetaTrader5 as mt5  # type: ignore

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
        import MetaTrader5 as mt5  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
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


def load_history_config(
    sym: str, cfg: dict, root: Path, validate: bool = False
) -> pd.DataFrame:
    """Load history for ``sym`` using local files, URLs or APIs."""

    csv_path = root / "data" / f"{sym}_history.csv"
    pq_path = root / "data" / f"{sym}_history.parquet"
    if pq_path.exists():
        logger.info("Loading history for %s from %s", sym, pq_path)
        return load_history_parquet(pq_path, validate=validate)
    if csv_path.exists():
        logger.info("Loading history for %s from %s", sym, csv_path)
        return load_history(csv_path, validate=validate)

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
        if validate:
            from .validators import TICK_SCHEMA

            TICK_SCHEMA.validate(df, lazy=True)
        save_history_parquet(df, pq_path)
        _record_data_version(pq_path, sym)
        return df

    urls = cfg.get("data_urls", {}).get(sym)
    if urls:
        logger.info("Downloading history for %s from URLs", sym)
        df = load_history_from_urls(urls)
        if validate:
            from .validators import TICK_SCHEMA

            TICK_SCHEMA.validate(df, lazy=True)
        save_history_parquet(df, pq_path)
        _record_data_version(pq_path, sym)
        return df

    raise FileNotFoundError(f"No history found for {sym} and no data source configured")


def load_history(path: Path, validate: bool = False) -> pd.DataFrame:
    """Load historical tick data from CSV.

    Parameters
    ----------
    path : Path
        CSV file path.
    validate : bool, optional
        If True, validate the resulting dataframe against ``TICK_SCHEMA``.
    """

    logger.info("Loading CSV history from %s", path)
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d %H:%M:%S:%f")
    if validate:
        from .validators import TICK_SCHEMA

        TICK_SCHEMA.validate(df, lazy=True)
    logger.debug("Loaded %d rows from CSV", len(df))
    _record_data_version(path)
    return df


def load_history_parquet(path: Path, validate: bool = False) -> pd.DataFrame:
    """Load historical tick data stored in a Parquet file.

    Parameters
    ----------
    path : Path
        Parquet file path.
    validate : bool, optional
        If True, validate the resulting dataframe against ``TICK_SCHEMA``.
    """

    logger.info("Loading Parquet history from %s", path)
    df = pd.read_parquet(path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)
    if validate:
        from .validators import TICK_SCHEMA

        TICK_SCHEMA.validate(df, lazy=True)
    logger.debug("Loaded %d rows from Parquet", len(df))
    _record_data_version(path)
    return df


def load_history_iter(path: Path, chunk_size: int):
    """Yield history dataframes from ``path`` in ``chunk_size`` rows.

    This utility streams Parquet data using ``pyarrow.dataset`` if available,
    falling back to ``pandas.read_parquet`` with ``chunksize``. Each yielded
    chunk has the ``Timestamp`` column normalized to naive ``datetime`` objects
    for consistency with other loaders.
    """

    logger.info(
        "Streaming Parquet history from %s in chunks of %d", path, chunk_size
    )
    try:  # Prefer the pyarrow dataset API for efficient streaming
        import pyarrow.dataset as ds  # type: ignore

        dataset = ds.dataset(path)
        for batch in dataset.to_batches(batch_size=chunk_size):
            df = batch.to_pandas()
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(
                    df["Timestamp"], utc=True
                ).dt.tz_localize(None)
            yield df
        return
    except Exception:  # pragma: no cover - pyarrow.dataset may be unavailable
        pass

    # Fallback to pandas iterator which also yields chunks
    for df in pd.read_parquet(path, chunksize=chunk_size):
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(
                None
            )
        yield df


def save_history_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save tick history to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True).dt.tz_localize(None)
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

__all__ = [
    "load_history_from_urls",
    "load_history_mt5",
    "load_history_config",
    "load_history",
    "load_history_parquet",
    "load_history_iter",
    "save_history_parquet",
    "load_multiple_histories",
]
