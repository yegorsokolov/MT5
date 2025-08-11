from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd
import argparse

from event_store import EventStore


def replay_event_log(path: str | Path) -> Dict[str, pd.DataFrame]:
    """Reconstruct DataFrames for each event type from the event log."""
    store = EventStore(path)
    features: list[dict] = []
    predictions: list[dict] = []
    orders: list[dict] = []
    fills: list[dict] = []
    for event in store.iter_events():
        et = event["type"]
        payload = event["payload"]
        if et == "feature":
            features.append(payload)
        elif et == "prediction":
            predictions.append(payload)
        elif et == "order":
            orders.append(payload)
        elif et == "fill":
            fills.append(payload)
    return {
        "features": pd.DataFrame(features),
        "predictions": pd.DataFrame(predictions),
        "orders": pd.DataFrame(orders),
        "fills": pd.DataFrame(fills),
    }


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Replay events from the event store")
    parser.add_argument(
        "path",
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "data" / "events.db",
        help="Path to event store database",
    )
    args = parser.parse_args()
    results = replay_event_log(args.path)
    for key, df in results.items():
        print(f"{key}: {len(df)} records")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
