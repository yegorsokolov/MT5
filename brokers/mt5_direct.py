import datetime as dt
import logging
from typing import List

import MetaTrader5 as _mt5
from utils.data_backend import get_dataframe_module

pd = get_dataframe_module()
import pandas as _pd
IS_CUDF = pd.__name__ == "cudf"


logger = logging.getLogger(__name__)

# Expose key constants for consumers
COPY_TICKS_ALL = _mt5.COPY_TICKS_ALL
ORDER_TYPE_BUY = _mt5.ORDER_TYPE_BUY
ORDER_TYPE_SELL = _mt5.ORDER_TYPE_SELL
TRADE_ACTION_DEAL = _mt5.TRADE_ACTION_DEAL

# Re-export utility functions used elsewhere
symbol_info_tick = _mt5.symbol_info_tick


def _to_datetime(arg, *args, **kwargs):
    func = getattr(pd, "to_datetime", _pd.to_datetime)
    result = func(arg, *args, **kwargs)
    if IS_CUDF and isinstance(result, _pd.Series):
        return pd.Series(result)
    return result


def _tz_localize_none(series):
    if IS_CUDF:
        return pd.Series(series.to_pandas().dt.tz_localize(None))
    return series.dt.tz_localize(None)


def _find_mt5_symbol(symbol: str):
    info = _mt5.symbol_info(symbol)
    if info:
        return symbol

    all_symbols = _mt5.symbols_get()
    for s in all_symbols:
        name = s.name
        if name.endswith(symbol) or name.startswith(symbol):
            return name
    return None


def initialize(**kwargs) -> bool:
    """Initialise MetaTrader5 connection.

    Any keyword arguments are passed directly to ``MetaTrader5.initialize``.
    The function returns ``True`` on success, ``False`` otherwise.
    """
    return bool(_mt5.initialize(**kwargs))


def order_send(request):
    """Proxy to ``MetaTrader5.order_send``."""
    return _mt5.order_send(request)


def symbol_select(symbol: str, enable: bool = True) -> bool:
    """Select or deselect a trading symbol."""
    return bool(_mt5.symbol_select(symbol, enable))


def positions_get(**kwargs):
    """Return open positions via ``MetaTrader5.positions_get``."""
    return _mt5.positions_get(**kwargs)


def copy_ticks_from(symbol: str, from_time, count: int, flags: int):
    """Proxy to ``MetaTrader5.copy_ticks_from``."""
    return _mt5.copy_ticks_from(symbol, from_time, count, flags)


def fetch_history(symbol: str, start: dt.datetime, end: dt.datetime):
    """Return tick history for ``symbol`` between ``start`` and ``end``.

    The function automatically selects the symbol if needed and requests data
    in week-long chunks to respect server limits.
    """

    if not initialize():  # pragma: no cover - optional dependency
        logger.error("Failed to initialize MetaTrader5")
        raise RuntimeError("Failed to initialize MetaTrader5")

    real_sym = _find_mt5_symbol(symbol)
    if not real_sym:
        _mt5.shutdown()
        raise ValueError(f"Symbol {symbol} not found in MetaTrader5")

    symbol_select(real_sym, True)

    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    chunk = 86400 * 7  # one week per request

    ticks: List = []
    cur = start_ts
    while cur < end_ts:
        to = min(cur + chunk, end_ts)
        arr = _mt5.copy_ticks_range(real_sym, cur, to, _mt5.COPY_TICKS_ALL)
        if arr is not None and len(arr) > 0:
            ticks.extend(arr)
        cur = to

    _mt5.shutdown()

    if not ticks:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)
    df["Timestamp"] = _tz_localize_none(
        _to_datetime(df["time"], unit="s", utc=True)
    )
    df.rename(columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True)
    df = df[["Timestamp", "Bid", "Ask", "Volume"]]
    df["BidVolume"] = df["Volume"]
    df["AskVolume"] = df["Volume"]
    df.drop(columns=["Volume"], inplace=True)

    logger.info("Loaded %d ticks from MetaTrader5", len(df))
    return df
