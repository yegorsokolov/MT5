from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import great_expectations as ge
from great_expectations.core.expectation_suite import ExpectationSuite

EXPECTATION_DIR = Path(__file__).resolve().parent
REPORT_DIR = Path("reports/data_quality")


def validate_dataframe(df: pd.DataFrame, suite_name: str, quarantine: bool = True) -> None:
    """Validate ``df`` against a Great Expectations suite.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate.
    suite_name : str
        Name of the expectation suite file (without extension).
    quarantine : bool, optional
        If True, failing dataframes are written to ``reports/data_quality/quarantine``
        before raising ``ValueError``.
    """

    suite_path = EXPECTATION_DIR / f"{suite_name}.json"
    with suite_path.open() as f:
        suite_dict = json.load(f)
    suite = ExpectationSuite(
        expectation_suite_name=suite_dict.get("expectation_suite_name", suite_name),
        expectations=suite_dict.get("expectations", []),
    )
    ge_df = ge.dataset.PandasDataset(df.copy())
    result = ge_df.validate(expectation_suite=suite)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    report_file = REPORT_DIR / f"{suite_name}_{ts}.json"
    with report_file.open("w") as f:
        json.dump(result.to_json_dict(), f, indent=2)

    if not result["success"]:
        if quarantine:
            qdir = REPORT_DIR / "quarantine"
            qdir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(qdir / f"{suite_name}_{ts}.parquet")
        raise ValueError(f"{suite_name} expectations failed; see {report_file}")
