import datetime as dt
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping

from utils.mt5_bridge import MetaTraderImportError, describe_backend, load_mt5_module

try:
    _mt5 = load_mt5_module()
except MetaTraderImportError as exc:  # pragma: no cover - surfaced during runtime checks
    raise ImportError(str(exc)) from exc
try:  # allow import to be stubbed in tests
    from utils.data_backend import get_dataframe_module
except Exception:  # pragma: no cover - fallback for tests
    import pandas as _pd  # type: ignore

    def get_dataframe_module():  # type: ignore
        return _pd

pd = get_dataframe_module()
import pandas as _pd
IS_CUDF = pd.__name__ == "cudf"


logger = logging.getLogger(__name__)

_BRIDGE_DETAILS = describe_backend()
if _BRIDGE_DETAILS:
    logger.debug("MetaTrader5 backend details: %s", _BRIDGE_DETAILS)

_INITIALIZED = False
_INITIALIZE_KWARGS = {
    "login",
    "password",
    "server",
    "path",
    "portable",
    "timeout",
    "timeout_ms",
    "timeout_sec",
    "host",
    "port",
}

from analysis.broker_tca import broker_tca


class MT5Error(RuntimeError):
    """Base exception raised for MetaTrader5 related failures."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class MT5OrderError(MT5Error):
    """Raised when an order could not be executed successfully."""


_SUCCESS_RETCODE_NAMES = (
    "TRADE_RETCODE_DONE",
    "TRADE_RETCODE_DONE_PARTIAL",
    "TRADE_RETCODE_PLACED",
    "TRADE_RETCODE_CANCELED",
    "TRADE_RETCODE_SL",
    "TRADE_RETCODE_TP",
)
_SUCCESS_RETCODES = {
    getattr(_mt5, name)
    for name in _SUCCESS_RETCODE_NAMES
    if hasattr(_mt5, name)
}
if not _SUCCESS_RETCODES:
    # ``MetaTrader5`` stubs used in unit tests may not expose the symbolic
    # constants.  Fall back to treating ``0`` as a generic success code.
    _SUCCESS_RETCODES = {0}


def _get_last_error() -> Tuple[Optional[int], Optional[str]]:
    """Return the latest MetaTrader5 error tuple if available."""

    last_error = getattr(_mt5, "last_error", None)
    if callable(last_error):
        try:
            code, message = last_error()
            if code == 0 and not message:
                return None, None
            return int(code), str(message)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to read MetaTrader5.last_error", exc_info=True)
    return None, None


def _compose_error_message(prefix: str, code: Optional[int], message: Optional[str]) -> str:
    if code is not None and message:
        return f"{prefix}: [{code}] {message}"
    if code is not None:
        return f"{prefix}: [{code}]"
    if message:
        return f"{prefix}: {message}"
    return prefix


def _raise_last_error(prefix: str) -> None:
    code, message = _get_last_error()
    err = MT5Error(_compose_error_message(prefix, code, message), code=code)
    raise _attach_resolution(err)


def _reset_initialized() -> None:
    global _INITIALIZED
    _INITIALIZED = False


def shutdown() -> bool:
    """Shut down the MetaTrader5 connection and reset initialisation state."""

    try:
        result = bool(_mt5.shutdown())
    except Exception:
        result = False
    finally:
        _reset_initialized()
    return result


def _ensure_initialized(
    *,
    login: Optional[int | str] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Initialise the MetaTrader5 backend if it is not already connected."""

    global _INITIALIZED
    if _INITIALIZED:
        return

    candidates: List[Optional[Dict[str, Any]]] = []

    if isinstance(config, Mapping):
        cfg_kwargs = {
            k: v
            for k, v in config.items()
            if v is not None and k in _INITIALIZE_KWARGS
        }
        if cfg_kwargs:
            candidates.append(cfg_kwargs)

    explicit_kwargs = {
        key: value
        for key, value in {"login": login, "password": password, "server": server}.items()
        if value is not None
    }
    if explicit_kwargs:
        candidates.append(explicit_kwargs)

    candidates.append(None)

    last_exc: Optional[Exception] = None
    for candidate in candidates:
        try:
            if candidate is None:
                success = bool(_mt5.initialize())
            else:
                success = bool(_mt5.initialize(**candidate))
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:  # pragma: no cover - defensive
            last_exc = exc
            continue

        if success:
            _INITIALIZED = True
            return

    code, message = _get_last_error()
    detail = _compose_error_message("MetaTrader5 initialize() failed", code, message)
    raise RuntimeError(detail) from last_exc


def _attach_resolution(error: "MT5Error") -> "MT5Error":
    """Populate ``error.details['resolution']`` with suggested actions."""

    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        error.details = {}  # type: ignore[attr-defined]
    try:  # pragma: no cover - defensive best effort
        from brokers.mt5_issue_solver import MT5IssueSolver

        solver = MT5IssueSolver()
        plan = solver.explain(error)
        if plan:
            error.details["resolution"] = plan  # type: ignore[index]
    except Exception:
        logger.debug("Failed to build MT5 resolution plan", exc_info=True)
    return error


def _timestamp_utcnow():
    for module in (pd, _pd):
        ts_cls = getattr(module, "Timestamp", None)
        utcnow = getattr(ts_cls, "utcnow", None) if ts_cls is not None else None
        if callable(utcnow):
            try:
                return utcnow()
            except Exception:
                continue
    return dt.datetime.now(dt.UTC)


def _ensure_order_success(result: Any, request: Dict[str, Any]) -> None:
    """Validate the response from ``MetaTrader5.order_send``."""

    if result is None:
        code, message = _get_last_error()
        details: Dict[str, Any] = {"request": request}
        err = MT5OrderError(
            _compose_error_message("MetaTrader5 order_send returned no result", code, message),
            code=code,
            details=details,
        )
        raise _attach_resolution(err)

    retcode = getattr(result, "retcode", None)
    if retcode in _SUCCESS_RETCODES or retcode is None:
        return

    comment = getattr(result, "comment", None)
    code, message = _get_last_error()
    msg_parts = [f"MetaTrader5 order_send failed with retcode {retcode}"]
    if comment:
        msg_parts.append(comment)
    if code is not None or message:
        msg_parts.append(_compose_error_message("last error", code, message))

    details = {
        "retcode": retcode,
        "comment": comment,
        "request_id": getattr(result, "request_id", None),
        "order": getattr(result, "order", None),
        "deal": getattr(result, "deal", None),
        "volume": getattr(result, "volume", None),
        "last_error_code": code,
        "last_error_message": message,
        "request": request,
    }
    raise _attach_resolution(
        MT5OrderError("; ".join(msg_parts), code=retcode, details=details)
    )

# Expose key constants for consumers
COPY_TICKS_ALL = getattr(_mt5, "COPY_TICKS_ALL", 0)
ORDER_TYPE_BUY = getattr(_mt5, "ORDER_TYPE_BUY", 0)
ORDER_TYPE_SELL = getattr(_mt5, "ORDER_TYPE_SELL", 0)
TRADE_ACTION_DEAL = getattr(_mt5, "TRADE_ACTION_DEAL", 0)

# Re-export utility functions used elsewhere
symbol_info_tick = getattr(_mt5, "symbol_info_tick", lambda *a, **k: None)


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


def _tick_to_dict(tick: Any) -> Dict[str, Any]:
    """Return a dictionary representation of ``tick``."""

    if isinstance(tick, dict):
        return dict(tick)

    data: Dict[str, Any] = {}

    # Named tuples expose ``_asdict`` for conversion.
    asdict = getattr(tick, "_asdict", None)
    if callable(asdict):
        data.update(asdict())

    dtype = getattr(getattr(tick, "dtype", None), "names", None)
    if dtype:
        try:
            data.update({name: tick[name] for name in dtype})
        except Exception:
            pass

    for key in ("time_msc", "time", "bid", "ask", "last", "volume", "volume_real", "flags"):
        if key in data:
            continue
        if hasattr(tick, key):
            try:
                data[key] = getattr(tick, key)
            except Exception:
                continue

    if not data and isinstance(tick, Iterable):
        keys = ["time", "bid", "ask", "volume"]
        for idx, key in enumerate(keys):
            try:
                data[key] = tick[idx]  # type: ignore[index]
            except Exception:
                break

    return data


def _ticks_to_dataframe(ticks: Iterable[Any]) -> _pd.DataFrame:
    """Convert tick structures to a pandas DataFrame."""

    rows = [_tick_to_dict(tick) for tick in ticks]
    if not rows:
        return _pd.DataFrame()
    return _pd.DataFrame(rows)


def _extract_symbol_name(candidate: Any) -> Optional[str]:
    """Best effort extraction of a symbol name from MetaTrader responses."""

    for attr in ("name", "symbol"):
        value = getattr(candidate, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(candidate, dict):
        for key in ("name", "symbol", "path"):
            value = candidate.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(candidate, (list, tuple)) and candidate:
        for item in candidate:
            if isinstance(item, str) and item:
                return item
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _iter_all_symbol_names() -> Iterable[str]:
    """Yield known symbol names from MetaTrader, handling API quirks."""

    symbols_get = getattr(_mt5, "symbols_get", None)
    symbols_total = getattr(_mt5, "symbols_total", None)
    seen: set[str] = set()

    def emit(values: Any) -> Iterable[str]:
        if not values:
            return []
        results: list[str] = []
        for entry in values:
            name = _extract_symbol_name(entry)
            if not name or name in seen:
                continue
            seen.add(name)
            results.append(name)
        return results

    if callable(symbols_get) and callable(symbols_total):
        try:
            total = int(symbols_total())
        except Exception:
            total = 0
        if total > 0:
            try:
                for name in emit(symbols_get(0, total)):
                    yield name
            except TypeError:
                # Older MetaTrader builds expose ``symbols_get()`` without
                # slice arguments – fall back to the parameterless call below.
                pass
            except Exception:
                pass

    if callable(symbols_get):
        fetchers = ((), (0,), None)
        for args in fetchers:
            try:
                if args is None:
                    values = symbols_get()
                else:
                    values = symbols_get(*args)
            except TypeError:
                continue
            except Exception:
                continue
            else:
                for name in emit(values):
                    yield name
                break


def _find_mt5_symbol(symbol: str):
    _ensure_initialized()
    info = _mt5.symbol_info(symbol)
    if info:
        return symbol

    for name in _iter_all_symbol_names():
        if name.endswith(symbol) or name.startswith(symbol):
            return name
    return None


def initialize(**kwargs) -> bool:
    """Initialise MetaTrader5 connection.

    Any keyword arguments are passed directly to ``MetaTrader5.initialize``.
    The function returns ``True`` on success, ``False`` otherwise.
    """
    # Always tear down existing connections so that switching accounts on the
    # terminal doesn't leave us bound to a stale session.
    shutdown()
    try:
        success = bool(_mt5.initialize(**kwargs))
    except Exception:
        success = False
    if not success:
        code, message = _get_last_error()
        logger.error(
            _compose_error_message("Failed to initialize MetaTrader5", code, message)
        )
        _reset_initialized()
        return False

    global _INITIALIZED
    _INITIALIZED = True
    return True


def is_terminal_logged_in() -> bool:
    """Return ``True`` if the MetaTrader5 terminal is logged in."""
    try:
        _ensure_initialized()
        info = _mt5.account_info()
        shutdown()
        return info is not None
    except RuntimeError:
        return False
    except Exception:
        shutdown()
        return False


def order_send(request):
    """Proxy to ``MetaTrader5.order_send`` with latency/slippage tracking."""

    order_ts = _timestamp_utcnow()
    try:
        result = _mt5.order_send(request)
    except Exception as exc:
        code, message = _get_last_error()
        details = {"request": request}
        err = MT5OrderError(
            _compose_error_message(
                "MetaTrader5 order_send raised an exception", code, message
            ),
            code=code,
            details=details,
        )
        raise _attach_resolution(err) from exc
    fill_ts = _timestamp_utcnow()
    try:
        req_price = float(request.get("price", 0) or 0)
        typ = request.get("type")
        fill_price = float(getattr(result, "price", req_price) or req_price)
        sign = 1 if typ in (ORDER_TYPE_BUY,) else -1
        slippage = (fill_price - req_price) / req_price * sign * 10000 if req_price else 0.0
        broker_tca.record("mt5_direct", order_ts, fill_ts, slippage)
    except Exception:
        pass
    _ensure_order_success(result, request)
    return result


def symbol_select(symbol: str, enable: bool = True) -> bool:
    """Select or deselect a trading symbol."""
    success = bool(_mt5.symbol_select(symbol, enable))
    if not success:
        action = "enable" if enable else "disable"
        _raise_last_error(f"Failed to {action} symbol {symbol}")
    return success


def positions_get(**kwargs):
    """Return open positions via ``MetaTrader5.positions_get``."""
    return _mt5.positions_get(**kwargs)


def copy_ticks_from(symbol: str, from_time, count: int, flags: int):
    """Proxy to ``MetaTrader5.copy_ticks_from``."""
    ticks = _mt5.copy_ticks_from(symbol, from_time, count, flags)
    if ticks is None:
        _raise_last_error(f"Failed to copy ticks for {symbol}")
    return ticks


def fetch_history(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    config: Optional[Mapping[str, Any]] = None,
    login: Optional[int | str] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
):
    """Return tick history for ``symbol`` between ``start`` and ``end``.

    The function automatically selects the symbol if needed and requests data
    in week-long chunks to respect server limits.
    """

    _ensure_initialized(
        config=config,
        login=login,
        password=password,
        server=server,
    )

    real_sym = _find_mt5_symbol(symbol)
    if not real_sym:
        shutdown()
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

    shutdown()

    if not ticks:
        return pd.DataFrame()

    df_pd = _ticks_to_dataframe(ticks)
    if df_pd.empty:
        return pd.DataFrame()

    if "time_msc" in df_pd.columns:
        time_col = "time_msc"
        unit = "ms"
    elif "time" in df_pd.columns:
        time_col = "time"
        unit = "s"
    else:
        raise KeyError("MetaTrader5 tick data missing 'time'/'time_msc' fields")

    if "bid" not in df_pd.columns and "last" in df_pd.columns:
        df_pd["bid"] = df_pd["last"]
    if "ask" not in df_pd.columns and "last" in df_pd.columns:
        df_pd["ask"] = df_pd["last"]
    if "volume" not in df_pd.columns and "volume_real" in df_pd.columns:
        df_pd["volume"] = df_pd["volume_real"]

    for col in ("bid", "ask", "volume"):
        if col not in df_pd.columns:
            df_pd[col] = 0

    df_pd["Timestamp"] = _tz_localize_none(
        _to_datetime(df_pd[time_col], unit=unit, utc=True)
    )
    df_pd.rename(columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True)

    df_pd = df_pd[["Timestamp", "Bid", "Ask", "Volume"]]
    df_pd["BidVolume"] = df_pd["Volume"]
    df_pd["AskVolume"] = df_pd["Volume"]
    df_pd.drop(columns=["Volume"], inplace=True)

    logger.info("Loaded %d ticks from MetaTrader5", len(df_pd))

    if pd is _pd:
        return df_pd

    try:
        dataframe_ctor = getattr(pd, "DataFrame", None)
        if dataframe_ctor is not None and hasattr(dataframe_ctor, "from_pandas"):
            return dataframe_ctor.from_pandas(df_pd)  # type: ignore[call-arg]
        if hasattr(pd, "from_pandas"):
            return pd.from_pandas(df_pd)  # type: ignore[attr-defined]
        return pd.DataFrame(df_pd)
    except Exception:
        return df_pd
