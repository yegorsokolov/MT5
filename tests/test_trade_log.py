from pathlib import Path
import importlib.util
import datetime as dt

spec = importlib.util.spec_from_file_location(
    "trade_log", Path(__file__).resolve().parents[1] / "data" / "trade_log.py"
)
trade_log = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trade_log)
TradeLog = trade_log.TradeLog


def test_logging_and_positions(tmp_path: Path):
    db = tmp_path / "trades.db"
    log = TradeLog(db)
    order_id = log.record_order(
        {
            "timestamp": dt.datetime(2024, 1, 1, 0, 0, 0),
            "symbol": "EURUSD",
            "side": "BUY",
            "volume": 1.0,
            "price": 1.1,
        }
    )
    log.record_fill(
        {
            "order_id": order_id,
            "timestamp": dt.datetime(2024, 1, 1, 0, 0, 1),
            "symbol": "EURUSD",
            "side": "BUY",
            "volume": 1.0,
            "price": 1.1,
        }
    )
    positions = log.get_open_positions()
    assert positions == [{"symbol": "EURUSD", "volume": 1.0, "avg_price": 1.1}]

    # restart recovery
    log2 = TradeLog(db)
    assert log2.get_open_positions() == positions
