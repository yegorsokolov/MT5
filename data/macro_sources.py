"""Client wrappers for macroeconomic data providers.

The platform historically relied on a small subset of public datasets.  This
module introduces first-class integrations with multiple macro data APIs so the
learning and execution models can react to richer context.  Each provider is
wrapped in a light-weight fetcher that normalises responses into a common
``Date``/``value`` dataframe shape.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional
from urllib.parse import parse_qs

import pandas as pd

try:  # pragma: no cover - optional dependency at runtime
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

logger = logging.getLogger(__name__)


class MacroDataError(RuntimeError):
    """Raised when a macro series cannot be resolved."""


@dataclass(frozen=True)
class SeriesSpec:
    """Describe a macro series request."""

    raw: str
    provider: str
    identifier: str
    params: Dict[str, str]
    alias: Optional[str] = None

    def column_name(self) -> str:
        """Column name to use for this series in joined dataframes."""

        if self.alias:
            return self.alias
        if "::" in self.raw:
            return self.identifier
        return self.raw


def parse_series_spec(symbol: str | SeriesSpec) -> SeriesSpec:
    """Return a :class:`SeriesSpec` parsed from ``symbol``.

    Symbols accept the ``provider::identifier?param=value`` syntax.  When no
    provider is specified FRED is assumed for backwards compatibility.  Query
    parameters are forwarded to the underlying API after removing the optional
    ``name`` alias parameter.
    """

    if isinstance(symbol, SeriesSpec):
        return symbol

    raw = symbol
    provider = "fred"
    remainder = symbol
    if "::" in symbol:
        provider, remainder = symbol.split("::", 1)
    provider = provider.strip().lower() or "fred"

    params: Dict[str, str] = {}
    identifier = remainder
    if "?" in remainder:
        identifier, query = remainder.split("?", 1)
        params = {
            key: values[-1]
            for key, values in parse_qs(query, keep_blank_values=True).items()
            if values
        }
    identifier = identifier.strip()
    alias = params.pop("name", params.pop("alias", None))
    return SeriesSpec(raw=raw, provider=provider, identifier=identifier, params=params, alias=alias)


def _records_to_frame(records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    records = list(records)
    if not records:
        return pd.DataFrame(columns=["Date", "value"])
    df = pd.DataFrame(records)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    return df.reset_index(drop=True)


def _fetch_fred(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = {"series_id": spec.identifier, "file_type": "json"}
    params.update({k: v for k, v in spec.params.items() if k not in {"api_key"}})
    api_key = spec.params.get("api_key") or os.getenv("FRED_API_KEY")
    if api_key:
        params["api_key"] = api_key
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end

    resp = client.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=20.0)
    resp.raise_for_status()
    data = resp.json()
    observations = data.get("observations", []) if isinstance(data, dict) else []
    records = []
    for obs in observations:
        date = obs.get("date") if isinstance(obs, dict) else None
        value = obs.get("value") if isinstance(obs, dict) else None
        if not date or value in (None, "", "."):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        records.append({"Date": date, "value": val})
    return _records_to_frame(records)


def _year_range(start: Optional[str], end: Optional[str]) -> Optional[str]:
    if not start and not end:
        return None
    start_year = start.split("-")[0] if start else ""
    end_year = end.split("-")[0] if end else ""
    if start_year or end_year:
        return f"{start_year}:{end_year}".strip(":")
    return None


def _fetch_worldbank(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    country = params.pop("country", params.pop("countries", "all"))
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{spec.identifier}"
    query = {"format": "json", "per_page": params.pop("per_page", "2000")}
    date_range = params.pop("date", None) or _year_range(start, end)
    if date_range:
        query["date"] = date_range
    query.update(params)

    resp = client.get(url, params=query, timeout=20.0)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        return pd.DataFrame(columns=["Date", "value"])
    entries = payload[1] or []
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        date = entry.get("date")
        value = entry.get("value")
        if not date or value in (None, ""):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        records.append({"Date": date, "value": val})
    return _records_to_frame(records)


def _fetch_imf(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    url = f"https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{spec.identifier}"
    resp = client.get(url, params=spec.params, timeout=20.0)
    resp.raise_for_status()
    payload = resp.json()
    dataset = payload.get("CompactData", {}).get("DataSet", {})
    series = dataset.get("Series")
    if isinstance(series, list):  # pragma: no branch - rare but possible
        series = series[0] if series else {}
    observations = series.get("Obs", []) if isinstance(series, dict) else []
    if isinstance(observations, dict):
        observations = [observations]
    records = []
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        date = obs.get("@TIME_PERIOD")
        value = obs.get("@OBS_VALUE")
        if not date or value in (None, ""):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        records.append({"Date": date, "value": val})
    return _records_to_frame(records)


def _fetch_dbnomics(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    params.setdefault("format", "json")
    params.setdefault("observations", "1")
    url = f"https://api.db.nomics.world/v22/series/{spec.identifier}"
    resp = client.get(url, params=params, timeout=20.0)
    resp.raise_for_status()
    payload = resp.json()
    docs = payload.get("series", {}).get("docs", []) if isinstance(payload, dict) else []
    if not docs:
        return pd.DataFrame(columns=["Date", "value"])
    series = docs[0]
    if not isinstance(series, dict):
        return pd.DataFrame(columns=["Date", "value"])
    periods = series.get("period_start_day") or series.get("period")
    values = series.get("value")
    if not periods or not values:
        return pd.DataFrame(columns=["Date", "value"])
    records = []
    for date, value in zip(periods, values):
        if value in (None, ""):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        records.append({"Date": date, "value": val})
    return _records_to_frame(records)


def _fetch_finnhub(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    token = params.pop("token", None) or os.getenv("FINNHUB_API_KEY")
    if not token:
        logger.debug("Finnhub token not configured – skipping %s", spec.identifier)
        return pd.DataFrame(columns=["Date", "value"])
    params.setdefault("symbol", spec.identifier)
    params["token"] = token
    url = "https://finnhub.io/api/v1/economic-data"
    resp = client.get(url, params=params, timeout=20.0)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        return pd.DataFrame(columns=["Date", "value"])
    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        ts = item.get("datetime")
        value = item.get("value")
        if ts is None or value in (None, ""):
            continue
        try:
            dt = pd.to_datetime(float(ts), unit="s", utc=True)
            val = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        records.append({"Date": dt, "value": val})
    return _records_to_frame(records)


def _fetch_alphavantage(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    api_key = (
        params.pop("apikey", None)
        or params.pop("api_key", None)
        or os.getenv("ALPHAVANTAGE_API_KEY")
    )
    if not api_key:
        logger.debug("Alpha Vantage API key missing – skipping %s", spec.identifier)
        return pd.DataFrame(columns=["Date", "value"])
    params.setdefault("function", spec.identifier)
    params["apikey"] = api_key
    params.setdefault("datatype", "json")
    url = "https://www.alphavantage.co/query"
    resp = client.get(url, params=params, timeout=20.0)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data")
    if isinstance(data, dict):
        data = data.get("data")
    if data is None:
        # some endpoints respond with "series" or "historical"
        data = payload.get("series") or payload.get("historical")
    if not isinstance(data, list):
        return pd.DataFrame(columns=["Date", "value"])
    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        date = item.get("date")
        value = item.get("value") or item.get("v")
        if not date or value in (None, "", "."):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        records.append({"Date": date, "value": val})
    return _records_to_frame(records)


def _fetch_eodhd(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    token = params.pop("api_token", None) or params.pop("token", None) or os.getenv("EODHD_API_TOKEN")
    if not token:
        logger.debug("EODHD token missing – skipping %s", spec.identifier)
        return pd.DataFrame(columns=["Date", "value"])
    params.setdefault("indicator", spec.identifier)
    params.setdefault("fmt", "json")
    params["api_token"] = token
    url = "https://eodhd.com/api/macroeconomics"
    resp = client.get(url, params=params, timeout=20.0)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return pd.DataFrame(columns=["Date", "value"])
    records = []
    for item in data:
        if not isinstance(item, dict):
            continue
        date = item.get("date")
        value = item.get("value")
        if not date or value in (None, "", "."):
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        records.append({"Date": date, "value": val})
    return _records_to_frame(records)


PROVIDERS = {
    "fred": _fetch_fred,
    "worldbank": _fetch_worldbank,
    "world_bank": _fetch_worldbank,
    "imf": _fetch_imf,
    "dbnomics": _fetch_dbnomics,
    "finnhub": _fetch_finnhub,
    "alphavantage": _fetch_alphavantage,
    "alpha_vantage": _fetch_alphavantage,
    "eodhd": _fetch_eodhd,
    "eodhistoricaldata": _fetch_eodhd,
}


def fetch_series_data(
    spec: SeriesSpec,
    client: "httpx.Client",
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a macro series described by ``spec`` using ``client``.

    Parameters
    ----------
    spec:
        Parsed series specification.
    client:
        Reusable HTTP client instance.
    start, end:
        Optional ISO date strings bounding the requested history.  Individual
        providers interpret these hints slightly differently (e.g. year ranges
        for the World Bank API).
    """

    fetcher = PROVIDERS.get(spec.provider)
    if fetcher is None:
        raise MacroDataError(f"Unknown macro provider '{spec.provider}'")
    try:
        df = fetcher(spec, client=client, start=start, end=end)
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning(
            "Failed to fetch macro series %s::%s: %s", spec.provider, spec.identifier, exc
        )
        return pd.DataFrame(columns=["Date", "value"])
    return df


__all__ = [
    "MacroDataError",
    "SeriesSpec",
    "fetch_series_data",
    "parse_series_spec",
]

