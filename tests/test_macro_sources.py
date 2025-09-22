from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

httpx = pytest.importorskip("httpx")

from data.macro_sources import fetch_series_data, parse_series_spec
from data.macro_features import load_macro_series

DATA_DIR = Path(__file__).parent / "data" / "macro_sources"


def _payload(name: str) -> dict | list:
    return json.loads((DATA_DIR / name).read_text())


def _mock_client(payload):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    return httpx.Client(transport=transport)


def test_parse_series_spec_alias_and_params():
    spec = parse_series_spec("fred::GDPC1?name=gdp&frequency=m")
    assert spec.provider == "fred"
    assert spec.identifier == "GDPC1"
    assert spec.alias == "gdp"
    assert spec.params == {"frequency": "m"}


def test_fetch_fred_series():
    client = _mock_client(_payload("fred_observations.json"))
    spec = parse_series_spec("fred::GDP")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(100.0), pytest.approx(101.5)]


def test_fetch_worldbank_series():
    client = _mock_client(_payload("worldbank_indicator.json"))
    spec = parse_series_spec("worldbank::NY.GDP.MKTP.CD?country=USA")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(25000000.0), pytest.approx(25500000.0)]


def test_fetch_imf_series():
    client = _mock_client(_payload("imf_compact.json"))
    spec = parse_series_spec("imf::IFS/A.US.PCPI_IX")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(98.1), pytest.approx(102.4)]


def test_fetch_dbnomics_series():
    client = _mock_client(_payload("dbnomics_series.json"))
    spec = parse_series_spec("dbnomics::ECB/EXR/A.USD.EUR.SP00.A")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(1.05), pytest.approx(1.08)]


def test_fetch_finnhub_series_requires_token(monkeypatch):
    client = _mock_client(_payload("finnhub_macro.json"))
    spec = parse_series_spec("finnhub::US.GDP?token=demo")
    df = fetch_series_data(spec, client)
    assert len(df) == 2
    assert pd.api.types.is_datetime64tz_dtype(df["Date"])  # ensure timestamps parsed


def test_fetch_alphavantage_series(monkeypatch):
    client = _mock_client(_payload("alphavantage_macro.json"))
    spec = parse_series_spec("alphavantage::REAL_GDP?apikey=test")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(2.5), pytest.approx(2.8)]


def test_fetch_eodhd_series():
    client = _mock_client(_payload("eodhd_macro.json"))
    spec = parse_series_spec("eodhd::gdp?api_token=test&country=USA")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(1.2), pytest.approx(1.3)]


def test_fetch_statcan_series():
    client = _mock_client(_payload("statcan_wds.json"))
    spec = parse_series_spec("statcan::v123456?latestN=2")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(100.1), pytest.approx(101.3)]


def test_fetch_bank_of_canada_series():
    client = _mock_client(_payload("bankofcanada_valet.json"))
    spec = parse_series_spec("bankofcanada::FXUSDCAD")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(1.35), pytest.approx(1.34)]


def test_fetch_open_canada_series():
    client = _mock_client(_payload("open_canada_datastore.json"))
    spec = parse_series_spec("open_canada::resource?date_field=REF_DATE&value_field=VALUE")
    df = fetch_series_data(spec, client, start="2023-02-01", end="2023-02-15")
    assert len(df) == 1
    assert df["value"].iloc[0] == pytest.approx(10.9)


def test_fetch_oecd_series():
    client = _mock_client(_payload("oecd_sdmx.json"))
    spec = parse_series_spec("oecd::MEI_CLI/CAN.CLI.A")
    df = fetch_series_data(spec, client)
    assert list(df["value"]) == [pytest.approx(98.1), pytest.approx(99.3)]


def test_load_macro_series_uses_remote_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    def handler(request: httpx.Request) -> httpx.Response:
        if "stlouisfed" in request.url.host:
            return httpx.Response(200, json=_payload("fred_observations.json"))
        if "alphavantage" in request.url.host:
            return httpx.Response(200, json=_payload("alphavantage_macro.json"))
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    df = load_macro_series(
        ["fred::GDP?name=gdp", "alphavantage::REAL_GDP?apikey=test&name=real_gdp"],
        session=client,
        start=datetime(2022, 1, 1),
        end=datetime(2023, 12, 31),
    )

    assert set(df.columns) == {"Date", "gdp", "real_gdp"}
    assert df["gdp"].iloc[0] == pytest.approx(100.0)
    assert (tmp_path / "data" / "macro").exists()

    # Second call should hit cache and work without HTTP access
    client.close()
    df_cached = load_macro_series(["fred::GDP?name=gdp"], session=None)
    assert not df_cached.empty
