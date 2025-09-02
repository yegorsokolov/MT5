import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.alt_data_loader as alt_loader
from data.features import add_alt_features
from data.validators import FEATURE_SCHEMA


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


def test_alt_data_loader_alignment(tmp_path, monkeypatch):
    _write_csv(
        tmp_path / "dataset" / "shipping" / "AAA.csv",
        ["Date", "shipping_metric"],
        [["2020-01-01", 1.2]],
    )
    _write_csv(
        tmp_path / "dataset" / "retail" / "AAA.csv",
        ["Date", "retail_sales"],
        [["2020-01-01", 10]],
    )
    _write_csv(
        tmp_path / "dataset" / "weather" / "AAA.csv",
        ["Date", "temperature"],
        [["2020-01-01", 25.0]],
    )

    monkeypatch.chdir(tmp_path)

    alt = alt_loader.load_alt_data(["AAA"])
    assert set(["shipping_metric", "retail_sales", "temperature"]).issubset(alt.columns)
    assert (alt["Symbol"] == "AAA").all()
    assert alt["Date"].dtype.tz is not None

    base = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2020-01-02"], utc=True),
            "Symbol": ["AAA"],
            "return": [0.0],
            "ma_5": [0.0],
            "ma_10": [0.0],
            "ma_30": [0.0],
            "ma_60": [0.0],
            "volatility_30": [0.0],
            "rsi_14": [0.0],
            "market_regime": [0],
        }
    )

    out = add_alt_features(base)

    assert out.loc[0, "shipping_metric"] == 1.2
    assert out.loc[0, "retail_sales"] == 10
    assert out.loc[0, "temperature"] == 25.0
    FEATURE_SCHEMA.validate(out)
