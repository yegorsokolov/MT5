from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Provide a minimal ``requests`` stub so importing the ``data`` package does not
# require the optional dependency in test environments.
requests_stub = types.ModuleType("requests")


def _dummy_get(*_args, **_kwargs):
    class _Response:
        def json(self):
            return []

        @property
        def text(self):
            return ""

    return _Response()


requests_stub.get = _dummy_get
sys.modules.setdefault("requests", requests_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    pass


class _ValidationError(Exception):
    pass


pydantic_stub.BaseModel = _BaseModel
pydantic_stub.ValidationError = _ValidationError
pydantic_stub.ConfigDict = dict
pydantic_stub.Field = lambda *args, **kwargs: None
pydantic_stub.field_validator = lambda *args, **kwargs: (lambda func: func)
pydantic_stub.model_validator = lambda *args, **kwargs: (lambda func: func)
sys.modules.setdefault("pydantic", pydantic_stub)

filelock_stub = types.ModuleType("filelock")


class _FileLock:
    def __init__(self, *args, **kwargs):
        pass

    def acquire(self, *args, **kwargs):  # pragma: no cover - stub behaviour
        return True

    def release(self, *args, **kwargs):  # pragma: no cover - stub behaviour
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


filelock_stub.FileLock = _FileLock
sys.modules.setdefault("filelock", filelock_stub)

psutil_stub = types.ModuleType("psutil")
psutil_stub.virtual_memory = lambda: types.SimpleNamespace(total=0, available=0)
psutil_stub.cpu_count = lambda *args, **kwargs: 1
psutil_stub.cpu_percent = lambda *args, **kwargs: 0.0
psutil_stub.disk_usage = lambda path: types.SimpleNamespace(total=0, used=0, free=0)
psutil_stub.net_io_counters = lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0)
sys.modules.setdefault("psutil", psutil_stub)

ge_stub = types.ModuleType("great_expectations")


class _FakeResult(dict):
    def to_json_dict(self):  # pragma: no cover - stub output
        return dict(self)


class _PandasDataset:
    def __init__(self, df):
        self.df = df

    def validate(self, expectation_suite=None):
        return _FakeResult(success=True)


ge_stub.dataset = types.SimpleNamespace(PandasDataset=_PandasDataset)
ge_core_stub = types.ModuleType("great_expectations.core")
ge_expectation_suite_stub = types.ModuleType("great_expectations.core.expectation_suite")


class _ExpectationSuite:
    def __init__(self, expectation_suite_name: str, expectations: list):
        self.expectation_suite_name = expectation_suite_name
        self.expectations = expectations


ge_expectation_suite_stub.ExpectationSuite = _ExpectationSuite
ge_core_stub.expectation_suite = ge_expectation_suite_stub
sys.modules.setdefault("great_expectations.core.expectation_suite", ge_expectation_suite_stub)
sys.modules.setdefault("great_expectations.core", ge_core_stub)
sys.modules.setdefault("great_expectations", ge_stub)

joblib_stub = sys.modules.get("joblib")
if joblib_stub is None:
    joblib_stub = types.ModuleType("joblib")
    sys.modules["joblib"] = joblib_stub


class _Memory:
    def __init__(self, *args, **kwargs):
        pass

    def cache(self, func, ignore=None):  # pragma: no cover - stub
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _wrapper.clear = lambda: None
        return _wrapper


joblib_stub.Memory = _Memory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracles.oracle_scalper import MetaculusClient, OracleScalper, PolymarketClient
import importlib.util

oracle_module_path = PROJECT_ROOT / "features" / "oracle_intelligence.py"
spec = importlib.util.spec_from_file_location("oracle_intelligence_test", oracle_module_path)
assert spec is not None and spec.loader is not None
oracle_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_module)
oracle_compute = oracle_module.compute


POLYMARKET_SAMPLE = [
    {
        "id": "pm1",
        "question": "Will Bitcoin close above $100k by 2025?",
        "slug": "bitcoin-100k-2025",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.6", "0.4"],
        "liquidityNum": "15000",
        "volumeNum": "500000",
        "endDate": "2025-12-31T23:59:59Z",
        "bestBid": "0.58",
        "bestAsk": "0.62",
    }
]


METACULUS_SAMPLE = [
    {
        "id": 42,
        "title": "Will Bitcoin market cap exceed $3T by 2025?",
        "forecasts_count": 120,
        "question": {
            "description": "Community forecast about Bitcoin adoption.",
            "aggregations": {
                "recency_weighted": {
                    "history": [
                        {
                            "centers": [0.55],
                            "interval_lower_bounds": [0.45],
                            "interval_upper_bounds": [0.65],
                            "forecaster_count": 200,
                            "end_time": 1735689600.0,
                        }
                    ]
                }
            },
            "scheduled_close_time": "2025-12-31T00:00:00Z",
        },
    }
]


def _polymarket_fetcher(limit: int):
    return POLYMARKET_SAMPLE


def _metaculus_fetcher(limit: int):
    return METACULUS_SAMPLE


def _build_scalper() -> OracleScalper:
    poly = PolymarketClient(fetcher=_polymarket_fetcher)
    meta = MetaculusClient(fetcher=_metaculus_fetcher)
    return OracleScalper(polymarket=poly, metaculus=meta)


def test_oracle_scalper_assess_probabilities():
    scalper = _build_scalper()
    events = scalper.collect(["BTC"], aliases={"BTC": ["Bitcoin"]})
    assert not events.empty

    summary = scalper.assess_probabilities(events)
    assert {"symbol", "oracle", "prob_weighted", "event_count"}.issubset(summary.columns)

    poly_row = summary.loc[summary["oracle"] == "polymarket"].iloc[0]
    assert pytest.approx(poly_row["prob_weighted"], rel=1e-5) == 0.6
    assert poly_row["event_count"] == 1

    meta_row = summary.loc[summary["oracle"] == "metaculus"].iloc[0]
    assert pytest.approx(meta_row["prob_weighted"], rel=1e-5) == 0.55
    assert meta_row["event_count"] == 1


def test_oracle_feature_module(monkeypatch):
    scalper = _build_scalper()
    monkeypatch.setattr(oracle_module, "OracleScalper", lambda: scalper)

    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
            "Symbol": ["BTC", "BTC", "BTC"],
            "price": [1.0, 1.1, 1.2],
        }
    )
    df.attrs["oracle_aliases"] = {"BTC": ["Bitcoin"]}

    augmented = oracle_compute(df)

    required_cols = {
        "polymarket_prob_mean",
        "metaculus_prob_mean",
        "oracle_prob_mean",
        "oracle_event_count",
    }
    assert required_cols.issubset(augmented.columns)
    assert augmented["polymarket_prob_mean"].notna().all()
    assert augmented["metaculus_prob_mean"].notna().all()

    # The combined oracle probability should be the average of individual probabilities.
    expected = (0.6 + 0.55) / 2
    assert pytest.approx(augmented["oracle_prob_mean"].iloc[-1], rel=1e-5) == expected
