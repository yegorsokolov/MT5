import MetaTrader5 as _mt5

# Expose key constants for consumers
COPY_TICKS_ALL = _mt5.COPY_TICKS_ALL
ORDER_TYPE_BUY = _mt5.ORDER_TYPE_BUY
ORDER_TYPE_SELL = _mt5.ORDER_TYPE_SELL
TRADE_ACTION_DEAL = _mt5.TRADE_ACTION_DEAL

# Re-export utility functions used elsewhere
symbol_info_tick = _mt5.symbol_info_tick


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
