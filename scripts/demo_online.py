#!/usr/bin/env python3
"""Example usage of the incremental online model."""
from pathlib import Path
import duckdb
import joblib

# load model
repo = Path(__file__).resolve().parent.parent
model_path = repo / "models" / "online.joblib"
if not model_path.exists():
    raise SystemExit("Run train_online.py first to create online.joblib")

model, _ = joblib.load(model_path)

# fetch most recent feature row
db_path = repo / "data" / "realtime.duckdb"
conn = duckdb.connect(db_path.as_posix())
row = conn.execute(
    "SELECT * FROM features ORDER BY Timestamp DESC LIMIT 1"
).fetch_df().iloc[0]

features = {c: row[c] for c in row.index if c not in {"Timestamp", "Symbol"}}
prob = model.predict_proba_one(features).get(1, 0.0)
print("Latest probability:", prob)
