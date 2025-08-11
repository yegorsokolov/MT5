from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from event_store import EventStore
from analysis.replay_events import replay_event_log


def test_replay_matches_records(tmp_path):
    db = tmp_path / "events.db"
    store = EventStore(db)
    features = [{"Timestamp": "2024-01-01T00:00:00", "value": 1}]
    preds = [{"Timestamp": "2024-01-01T00:00:01", "Symbol": "EURUSD", "prob": 0.7}]
    order = {"timestamp": "2024-01-01T00:00:02", "symbol": "EURUSD", "side": "BUY", "volume": 1, "price": 1.1}
    fill = {**order, "order_id": 1}
    for rec in features:
        store.record("feature", rec)
    for rec in preds:
        store.record("prediction", rec)
    store.record("order", order)
    store.record("fill", fill)
    result = replay_event_log(db)
    assert result["features"].to_dict("records") == features
    assert result["predictions"].to_dict("records") == preds
    assert result["orders"].to_dict("records")[0]["symbol"] == order["symbol"]
    assert result["fills"].to_dict("records")[0]["order_id"] == 1
