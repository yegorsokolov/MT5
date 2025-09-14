import sys
from pathlib import Path

import pandas as pd
import importlib.util

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "fundamental_features", ROOT / "data" / "fundamental_features.py"
)
ff = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ff)  # type: ignore[attr-defined]


def test_load_fundamentals_additional_metrics(monkeypatch):
    def fake_read(sym):
        return pd.DataFrame()

    def fake_fetch(sym):
        return pd.DataFrame(
            {
                "Date": ["2024-01-01"],
                "pe_ratio": [10],
                "dividend_yield": [0.01],
                "eps": [2.5],
                "revenue": [1000],
                "ebitda": [200],
                "market_cap": [1_000_000],
            }
        )

    monkeypatch.setattr(ff, "_read_local_csv", fake_read)
    monkeypatch.setattr(ff, "_fetch_yfinance", fake_fetch)
    df = ff.load_fundamentals(["AAA"])
    assert set(["eps", "revenue", "ebitda", "market_cap"]).issubset(df.columns)
    assert (df["eps"] == 2.5).all()


def test_compute_merges_additional_metrics(monkeypatch):
    base = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-02"], utc=True),
            "Symbol": ["AAA"],
        }
    )

    def fake_load(symbols):
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01"], utc=True),
                "Symbol": ["AAA"],
                "pe_ratio": [10],
                "dividend_yield": [0.01],
                "eps": [2.5],
                "revenue": [1000],
                "ebitda": [200],
                "market_cap": [1_000_000],
            }
        )

    monkeypatch.setattr(ff, "load_fundamentals", fake_load)
    out = ff.compute(base)
    for col in [
        "pe_ratio",
        "dividend_yield",
        "eps",
        "revenue",
        "ebitda",
        "market_cap",
    ]:
        assert col in out.columns
        assert not out[col].isna().any()
