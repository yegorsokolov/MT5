from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import argparse

from event_store import EventStore

try:  # optional pyarrow dependency
    import pyarrow.dataset as ds  # type: ignore
except Exception:  # pragma: no cover
    ds = None


def _replay_parquet(path: Path) -> Dict[str, pd.DataFrame]:
    if ds is None:
        return {"features": pd.DataFrame(), "predictions": pd.DataFrame(), "orders": pd.DataFrame(), "fills": pd.DataFrame()}
    dataset = ds.dataset(path, format="parquet", partitioning="hive")

    def _fetch(et: str) -> pd.DataFrame:
        try:
            table = dataset.to_table(filter=ds.field("type") == et)
            return table.to_pandas()
        except Exception:
            return pd.DataFrame()

    return {
        "features": _fetch("feature"),
        "predictions": _fetch("prediction"),
        "orders": _fetch("order"),
        "fills": _fetch("fill"),
    }


def replay_event_log(path: str | Path) -> Dict[str, pd.DataFrame]:
    """Reconstruct DataFrames for each event type from the event log."""
    path = Path(path)
    if path.is_file():
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
    else:
        return _replay_parquet(path)


def compare_model_outcomes(path: str | Path, model_loader: Any | None = None) -> pd.DataFrame:
    """Replay events and compare historical vs current model predictions."""
    data = replay_event_log(path)
    feats = data.get("features", pd.DataFrame())
    hist = data.get("predictions", pd.DataFrame())
    if feats.empty or hist.empty:
        return pd.DataFrame()
    if model_loader is None:
        try:  # lazy import to avoid heavy dependency if unused
            from generate_signals import load_models

            models, _ = load_models([])
            model = next(iter(models.values())) if models else None
        except Exception:
            model = None
    else:
        model = model_loader()
    if model is None or not hasattr(model, "predict"):
        current = pd.Series([None] * len(feats))
    else:
        try:
            current = pd.Series(model.predict(feats))
        except Exception:
            current = pd.Series([None] * len(feats))
    hist_series: pd.Series | pd.DataFrame | None = None
    if "pred" in hist:
        hist_series = hist["pred"]
    elif "prediction" in hist:
        val = hist["prediction"]
        hist_series = val.get("mean") if isinstance(val, pd.DataFrame) else val
    if hist_series is None:
        return pd.DataFrame()
    comp = pd.DataFrame({"historical": hist_series, "current": current[: len(hist_series)]})
    return comp


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Replay events from the event store")
    parser.add_argument(
        "path",
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "data" / "events.db",
        help="Path to event store database or dataset",
    )
    args = parser.parse_args()
    results = replay_event_log(args.path)
    for key, df in results.items():
        print(f"{key}: {len(df)} records")
    comp = compare_model_outcomes(args.path)
    if not comp.empty:
        diff = (comp["current"] - comp["historical"]).abs().mean()
        print(f"Mean absolute prediction diff: {diff}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
