import pandas as pd
import pytest
import runpy
from pathlib import Path

validate_ge = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "features" / "validators.py")
)["validate_ge"]
ingest = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "data" / "ingest.py"), run_name="__test__"
)["ingest"]


def test_validate_ge_success():
    df = pd.DataFrame({"price": [1, 2, 3]})
    validate_ge(df, "prices")


def test_validate_ge_failure():
    df = pd.DataFrame({"price": [1, None]})
    with pytest.raises(ValueError):
        validate_ge(df, "prices")


def test_ingest_uses_validation(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("price\n1\n2\n")
    ingest(good)
    bad = tmp_path / "bad.csv"
    bad.write_text("wrong\n1\n")
    with pytest.raises(ValueError):
        ingest(bad)
