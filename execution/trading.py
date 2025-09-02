"""High-level trading helpers using the active broker backend."""
from __future__ import annotations

from typing import Any, Optional

from brokers import connection_manager as conn_mgr


def place_order(symbol: str, side: str, volume: float, price: float | None = None) -> Any:
    """Place a market order for ``symbol`` via the active broker.

    Parameters
    ----------
    symbol: str
        Trading symbol (e.g. ``"EURUSD"``).
    side: str
        ``"buy"`` or ``"sell"``.
    volume: float
        Lot size to trade.
    price: float, optional
        Price to use; if ``None`` the current tick price is used.
    """
    broker = conn_mgr.get_active_broker()
    broker.symbol_select(symbol, True)
    tick = broker.symbol_info_tick(symbol)
    if price is None:
        price = tick.ask if side.lower() == "buy" else tick.bid
    request = {
        "action": broker.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": broker.ORDER_TYPE_BUY if side.lower() == "buy" else broker.ORDER_TYPE_SELL,
        "price": price,
    }
    return broker.order_send(request)


def close_position(position_id: int, volume: Optional[float] = None) -> Any:
    """Close an existing position by ticket id."""
    broker = conn_mgr.get_active_broker()
    pos = broker.positions_get(ticket=position_id)
    if not pos:
        return None
    pos = pos[0]
    symbol = pos.symbol
    volume = volume or pos.volume
    tick = broker.symbol_info_tick(symbol)
    side = broker.ORDER_TYPE_SELL if pos.type == broker.ORDER_TYPE_BUY else broker.ORDER_TYPE_BUY
    price = tick.bid if side == broker.ORDER_TYPE_SELL else tick.ask
    request = {
        "action": broker.TRADE_ACTION_DEAL,
        "position": position_id,
        "symbol": symbol,
        "volume": volume,
        "type": side,
        "price": price,
    }
    return broker.order_send(request)
