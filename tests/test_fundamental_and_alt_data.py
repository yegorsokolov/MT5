import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.resource_monitor import monitor
import data.features as feature_module
from data.features import make_features


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


def test_fundamental_and_alt_data_alignment(tmp_path, monkeypatch):
    # Prepare minimal dataset structure
    _write_csv(
        tmp_path / "dataset" / "fundamentals" / "AAA.csv",
        ["Date", "revenue", "net_income"],
        [["2020-01-01", 100, 10]],
    )
    _write_csv(
        tmp_path / "dataset" / "ratios" / "AAA.csv",
        ["Date", "pe_ratio", "dividend_yield"],
        [["2020-01-01", 15, 0.02]],
    )
    _write_csv(
        tmp_path / "dataset" / "options" / "AAA.csv",
        ["Date", "implied_vol"],
        [["2020-01-01", 0.3]],
    )
    _write_csv(
        tmp_path / "dataset" / "onchain" / "AAA.csv",
        ["Date", "active_addresses"],
        [["2020-01-01", 123]],
    )
    _write_csv(
        tmp_path / "dataset" / "esg" / "AAA.csv",
        ["Date", "esg_score"],
        [["2020-01-01", 80]],
    )
    # Macro series
    _write_csv(
        tmp_path / "dataset" / "gdp.csv",
        ["Date", "gdp"],
        [["2020-01-01", 2.0]],
    )
    _write_csv(
        tmp_path / "dataset" / "cpi.csv",
        ["Date", "cpi"],
        [["2020-01-01", 1.5]],
    )
    _write_csv(
        tmp_path / "dataset" / "interest_rate.csv",
        ["Date", "interest_rate"],
        [["2020-01-01", 0.5]],
    )

    monkeypatch.chdir(tmp_path)
    # ensure feature registry does not modify df
    monkeypatch.setattr(feature_module, "get_feature_pipeline", lambda: [lambda df: df])
    monitor.capability_tier = "hpc"

    base = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True),
            "Symbol": ["AAA", "AAA"],
            "return": [0.0, 0.0],
            "ma_5": [0.0, 0.0],
            "ma_10": [0.0, 0.0],
            "ma_30": [0.0, 0.0],
            "ma_60": [0.0, 0.0],
            "volatility_30": [0.0, 0.0],
            "rsi_14": [0.0, 0.0],
            "market_regime": [0, 0],
        }
    )

    out = make_features(base, validate=True)

    for col in [
        "revenue",
        "net_income",
        "pe_ratio",
        "dividend_yield",
        "gdp",
        "cpi",
        "interest_rate",
        "implied_vol",
        "active_addresses",
        "esg_score",
    ]:
        assert col in out.columns
        assert not out[col].isna().any()

    assert (out["pe_ratio"] == 15).all()
    assert (out["implied_vol"] == 0.3).all()
