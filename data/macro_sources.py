"""Client wrappers for macroeconomic data providers.

The platform historically relied on a small subset of public datasets.  This
module introduces first-class integrations with multiple macro data APIs so the
learning and execution models can react to richer context.  Each provider is
wrapped in a light-weight fetcher that normalises responses into a common
``Date``/``value`` dataframe shape.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional
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


def _statcan_period_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    if len(text) > 10:
        text = text[:10]
    return text


def _statcan_vector_points(payload: Any) -> list[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("vectorDataPoints", "vectorDataPoint"):
            data = payload.get(key)
            if isinstance(data, list):
                return [entry for entry in data if isinstance(entry, dict)]
        for key in ("object", "data", "result", "results"):
            nested = payload.get(key)
            points = _statcan_vector_points(nested)
            if points:
                return points
    return []


def _statcan_parse_records(data: Any) -> Iterable[Dict[str, object]]:
    points = _statcan_vector_points(data)
    for point in points:
        if not isinstance(point, dict):
            continue
        date = None
        for key in ("refPer2", "refPer", "referencePeriod", "refPeriod", "ref_date", "date"):
            if key in point and point[key]:
                date = point[key]
                break
        value = point.get("value") or point.get("val") or point.get("value_decimal")
        if date is None or value in (None, "", ".", ".."):
            continue
        try:
            numeric = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            continue
        yield {"Date": date, "value": numeric}


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


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _format_ckan_value(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, (list, tuple, set)):
        parts = ", ".join(_format_ckan_value(item) for item in value)
        return f"({parts})"
    return f"'{json.dumps(value)}'"


def _detect_date_field(records: Iterable[Dict[str, Any]], preferred: Optional[str]) -> Optional[str]:
    records = [r for r in records if isinstance(r, dict)]
    if not records:
        return preferred
    if preferred and all(preferred in r for r in records):
        return preferred
    for candidate in (preferred, "Date", "date", "ref_date", "REF_DATE", "reference_date"):
        if candidate and all(candidate in r for r in records):
            return candidate
    for key in records[0].keys():
        if "date" in key.lower():
            return key
    return preferred


def _is_number(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    try:
        float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return False
    return True


def _detect_value_field(records: Iterable[Dict[str, Any]], preferred: Optional[str], date_field: Optional[str]) -> Optional[str]:
    records = [r for r in records if isinstance(r, dict)]
    if not records:
        return preferred
    if preferred and all(preferred in r for r in records):
        return preferred
    candidates = []
    for record in records:
        for key, value in record.items():
            if key == date_field or key == "_id":
                continue
            if isinstance(value, dict):
                if _is_number(value.get("value")) or _is_number(value.get("v")):
                    return key
                continue
            if _is_number(value):
                candidates.append(key)
        if candidates:
            break
    if candidates:
        return candidates[0]
    for fallback in ("value", "VALUE", "val", "OBS_VALUE"):
        if all(fallback in r for r in records):
            return fallback
    return preferred


def _fetch_statcan(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop("base_url", "https://www150.statcan.gc.ca/t1/wds/en/grp/v1").rstrip("/")
    vector_id = params.pop("vector_id", params.pop("vector", None)) or spec.identifier
    vector_id = vector_id.strip()
    if not vector_id:
        return pd.DataFrame(columns=["Date", "value"])

    payload_param = params.pop("payload", None)
    if isinstance(payload_param, str):
        try:
            payload: Dict[str, Any] = json.loads(payload_param)
        except json.JSONDecodeError:
            payload = {}
    elif isinstance(payload_param, dict):
        payload = dict(payload_param)
    else:
        payload = {}

    method = params.pop("method", None)
    start_period = params.pop("startPeriod", params.pop("start_period", None))
    end_period = params.pop("endPeriod", params.pop("end_period", None))
    latest_n = params.pop("latest_n", params.pop("latestN", params.pop("latest", None)))

    if start and not start_period:
        start_period = _statcan_period_hint(start)
    if end and not end_period:
        end_period = _statcan_period_hint(end)

    query_params = {k: v for k, v in params.items() if v is not None}

    if method:
        endpoint = f"{base_url}/{method.lstrip('/')}"
        if "vectorId" not in payload and "vectorIds" not in payload:
            payload.setdefault("vectorId", vector_id)
    elif start_period or end_period:
        endpoint = f"{base_url}/getDataFromVectorAndPeriodRange"
        payload.setdefault("vectorId", vector_id)
        if start_period and "startPeriod" not in payload:
            payload["startPeriod"] = start_period
        if end_period and "endPeriod" not in payload:
            payload["endPeriod"] = end_period
    else:
        endpoint = f"{base_url}/getDataFromVectorsAndLatestNPeriods"
        vectors = payload.get("vectorIds")
        if not isinstance(vectors, list) or not vectors:
            payload["vectorIds"] = [vector_id]
        if latest_n and "latestN" not in payload:
            try:
                payload["latestN"] = int(latest_n)
            except (TypeError, ValueError):
                pass
        payload.setdefault("latestN", payload.get("latestN", 365))

    try:
        resp = client.post(endpoint, params=query_params or None, json=payload or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("Statistics Canada request failed for %s: %s", vector_id, exc)
        return pd.DataFrame(columns=["Date", "value"])

    payload_data = resp.json()
    records = list(_statcan_parse_records(payload_data))
    return _records_to_frame(records)


def _fetch_bank_of_canada(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop("base_url", "https://www.bankofcanada.ca/valet").rstrip("/")
    path = params.pop("path", None)
    series_key = params.pop("series", None)
    if path:
        resource = path.strip("/")
    else:
        identifier = spec.identifier.strip("/")
        if not identifier:
            return pd.DataFrame(columns=["Date", "value"])
        if identifier.startswith("observations"):
            resource = identifier
        else:
            resource = f"observations/{identifier}"
    if not resource.endswith(".json"):
        endpoint = f"{base_url}/{resource}.json"
    else:
        endpoint = f"{base_url}/{resource}"

    query: Dict[str, Any] = {k: v for k, v in params.items() if k not in {"start_date", "end_date"} and v is not None}
    if start and "start_date" not in params:
        query["start_date"] = start
    elif params.get("start_date"):
        query["start_date"] = params["start_date"]
    if end and "end_date" not in params:
        query["end_date"] = end
    elif params.get("end_date"):
        query["end_date"] = params["end_date"]

    try:
        resp = client.get(endpoint, params=query or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("Bank of Canada request failed for %s: %s", spec.identifier, exc)
        return pd.DataFrame(columns=["Date", "value"])

    payload = resp.json()
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return pd.DataFrame(columns=["Date", "value"])

    if not series_key:
        detail = payload.get("seriesDetail")
        if isinstance(detail, dict) and len(detail) == 1:
            series_key = next(iter(detail))
        elif observations:
            for key in observations[0].keys():
                if key.lower() not in {"d", "date"}:
                    series_key = key
                    break

    records = []
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        date = obs.get("d") or obs.get("date") or obs.get("DATE")
        if not date:
            continue
        value_obj: Any
        if series_key:
            value_obj = obs.get(series_key)
        else:
            value_obj = next((v for k, v in obs.items() if k.lower() not in {"d", "date"}), None)
        if isinstance(value_obj, dict):
            value = value_obj.get("v") or value_obj.get("value") or value_obj.get("value_decimal")
        else:
            value = value_obj
        if value in (None, ""):
            continue
        try:
            numeric = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            continue
        records.append({"Date": date, "value": numeric})
    return _records_to_frame(records)


def _fetch_open_canada(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop("base_url", "https://open.canada.ca/data/en/api").rstrip("/")
    resource_id = params.pop("resource_id", params.pop("resource", None)) or spec.identifier
    resource_id = resource_id.strip()
    if not resource_id:
        return pd.DataFrame(columns=["Date", "value"])

    date_field = params.pop("date_field", params.pop("date", None))
    value_field = params.pop("value_field", params.pop("column", params.pop("value", None)))
    filters = _coerce_mapping(params.pop("filters", None))
    limit = params.pop("limit", params.pop("rows", None))
    try:
        limit_int = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit_int = None

    sql = params.pop("sql", None)
    if isinstance(sql, str) and sql.strip():
        endpoint = f"{base_url}/action/datastore_search_sql"
        query_params: Dict[str, Any] = {"sql": sql}
    else:
        endpoint = f"{base_url}/action/datastore_search"
        query_params = {"resource_id": resource_id}
        if limit_int is not None:
            query_params["limit"] = limit_int
        if filters:
            query_params["filters"] = json.dumps(filters)
        for key in ("offset", "sort", "fields", "q"):
            if key in params:
                query_params[key] = params.pop(key)

        if date_field and (start or end):
            # Promote to SQL for range filtering
            columns = [date_field]
            if value_field and value_field != date_field:
                columns.append(value_field)
            sql_cols = ", ".join(f'"{col}"' for col in dict.fromkeys(columns))
            sql_query = f'SELECT {sql_cols} FROM "{resource_id}"'
            conditions = []
            for key, value in filters.items():
                conditions.append(f'"{key}" = {_format_ckan_value(value)}')
            if start:
                conditions.append(f'"{date_field}" >= {_format_ckan_value(start)}')
            if end:
                conditions.append(f'"{date_field}" <= {_format_ckan_value(end)}')
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            sql_query += f' ORDER BY "{date_field}"'
            if limit_int is not None:
                sql_query += f" LIMIT {limit_int}"
            endpoint = f"{base_url}/action/datastore_search_sql"
            query_params = {"sql": sql_query}

    if params:
        query_params.update({k: v for k, v in params.items() if v is not None})

    try:
        resp = client.get(endpoint, params=query_params or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("Open Canada request failed for %s: %s", resource_id, exc)
        return pd.DataFrame(columns=["Date", "value"])

    payload = resp.json()
    result = payload.get("result") if isinstance(payload, dict) else None
    records = []
    if isinstance(result, dict) and isinstance(result.get("records"), list):
        records = [r for r in result["records"] if isinstance(r, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = [r for r in payload["records"] if isinstance(r, dict)]
    if not records:
        return pd.DataFrame(columns=["Date", "value"])

    date_field = _detect_date_field(records, date_field)
    value_field = _detect_value_field(records, value_field, date_field)
    if not date_field and records:
        for key in records[0].keys():
            if key != "_id":
                date_field = key
                break

    detected_value_field = value_field
    out_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        date_value = record.get(date_field) if date_field else None
        if date_value in (None, ""):
            continue

        source_field = detected_value_field
        value_obj = record.get(source_field) if source_field else None
        if isinstance(value_obj, dict):
            value_obj = value_obj.get("value") or value_obj.get("v")

        if value_obj in (None, ""):
            source_field = None
            for key, candidate in record.items():
                if key in {date_field, "_id"}:
                    continue
                extracted = candidate
                if isinstance(candidate, dict):
                    extracted = candidate.get("value") or candidate.get("v")
                if extracted in (None, ""):
                    continue
                if _is_number(extracted):
                    value_obj = extracted
                    source_field = key
                    break

        if value_obj in (None, ""):
            continue

        try:
            numeric = float(str(value_obj).replace(",", ""))
        except (TypeError, ValueError):
            continue

        if not detected_value_field and source_field:
            detected_value_field = source_field
        out_records.append({"Date": date_value, "value": numeric})

    df = _records_to_frame(out_records)
    if df.empty:
        return df
    if start:
        start_ts = pd.to_datetime(start, utc=True, errors="coerce")
        if pd.notna(start_ts):
            df = df[df["Date"] >= start_ts]
    if end:
        end_ts = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.notna(end_ts):
            df = df[df["Date"] <= end_ts]
    return df.reset_index(drop=True)


def _sdmx_time_labels(structure: Any) -> list[str]:
    return _oecd_time_labels(structure)


def _sdmx_parse_observations(observations: Dict[str, Any], structure: Any) -> Iterable[Dict[str, object]]:
    return _parse_oecd_observations(observations, structure)


def _sdmx_collect_records(payload: Any) -> Iterable[Dict[str, object]]:
    if not isinstance(payload, dict):
        return []

    data_sets = payload.get("dataSets")
    dataset = None
    if isinstance(data_sets, list) and data_sets:
        dataset = data_sets[0]
    elif isinstance(data_sets, dict):
        dataset = data_sets.get("0") or next(iter(data_sets.values()), None)
    if not isinstance(dataset, dict):
        dataset = payload.get("dataSet")
    if not isinstance(dataset, dict):
        return []

    structure = payload.get("structure") if isinstance(payload, dict) else None

    records: list[Dict[str, object]] = []
    observations = dataset.get("observations")
    if isinstance(observations, dict):
        records.extend(_sdmx_parse_observations(observations, structure))

    series_map = dataset.get("series")
    if isinstance(series_map, dict):
        for series in series_map.values():
            if isinstance(series, dict) and isinstance(series.get("observations"), dict):
                records.extend(_sdmx_parse_observations(series["observations"], structure))

    if not records and isinstance(dataset.get("data"), dict):
        for series in dataset["data"].values():
            if isinstance(series, dict) and isinstance(series.get("observations"), dict):
                records.extend(_sdmx_parse_observations(series["observations"], structure))

    return records


def _ons_latest_version(
    client: "httpx.Client", base_url: str, dataset: str, edition: str
) -> Optional[str]:
    url = f"{base_url}/datasets/{dataset}/editions/{edition}/versions"
    try:
        resp = client.get(url, timeout=20.0)
        resp.raise_for_status()
    except Exception:  # pragma: no cover - depends on remote availability
        return None

    payload = resp.json()
    items = []
    if isinstance(payload, dict):
        candidates = payload.get("items") or payload.get("versions") or payload.get("results")
        if isinstance(candidates, list):
            items = [item for item in candidates if isinstance(item, dict)]

    best_version: Optional[str] = None
    best_numeric = -1
    for item in items:
        value = item.get("version") or item.get("id") or item.get("number")
        if value is None:
            continue
        text = str(value)
        try:
            numeric = int(text)
        except ValueError:
            numeric = best_numeric
        if numeric >= best_numeric:
            best_numeric = numeric
            best_version = text
    return best_version


def _ons_extract_dimension(option: Any) -> Optional[str]:
    if isinstance(option, dict):
        for key in ("id", "label", "value"):
            val = option.get(key)
            if val not in (None, ""):
                return str(val)
    if isinstance(option, str):
        return option
    return None


def _ons_extract_date(obs: Dict[str, Any], time_dimension: Optional[str]) -> Optional[str]:
    dimensions = obs.get("dimensions")
    if isinstance(dimensions, dict):
        if time_dimension:
            dim_meta = dimensions.get(time_dimension)
            if isinstance(dim_meta, dict):
                option = dim_meta.get("option")
                candidate = _ons_extract_dimension(option)
                if candidate:
                    return candidate
                candidate = dim_meta.get("label") or dim_meta.get("id")
                if candidate:
                    return str(candidate)
        for key, meta in dimensions.items():
            if not isinstance(meta, dict):
                continue
            key_lower = key.lower()
            meta_id = str(meta.get("id", "")).lower()
            if key_lower in {"time", "period", "date"} or meta_id in {"time", "period", "date"}:
                option = meta.get("option")
                candidate = _ons_extract_dimension(option)
                if candidate:
                    return candidate
                candidate = meta.get("label") or meta.get("name") or meta.get("id")
                if candidate:
                    return str(candidate)
        for meta in dimensions.values():
            if not isinstance(meta, dict):
                continue
            option = meta.get("option")
            candidate = _ons_extract_dimension(option)
            if candidate:
                return candidate
    for key in ("time", "date", "period", "referencePeriod", "ref_period", "observation_date"):
        value = obs.get(key)
        if value:
            return str(value)
    return None


def _fetch_ons(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop("base_url", "https://api.beta.ons.gov.uk/v1").rstrip("/")
    dataset = params.pop("dataset", params.pop("dataset_id", None)) or spec.identifier
    dataset = dataset.strip("/")
    if not dataset:
        return pd.DataFrame(columns=["Date", "value"])

    edition = params.pop("edition", params.pop("edition_id", "time-series"))
    edition = edition.strip("/") or "time-series"
    version = params.pop("version", params.pop("dataset_version", None))
    time_dimension = params.pop("time_dimension", params.pop("time_dim", None))
    value_field = params.pop("value_field", None)

    if not version or str(version).lower() == "latest":
        detected = _ons_latest_version(client, base_url, dataset, edition)
        if detected:
            version = detected

    version = str(version or "1").strip()
    endpoint = f"{base_url}/datasets/{dataset}/editions/{edition}/versions/{version}/observations"
    query = {k: v for k, v in params.items() if v not in (None, "")}

    try:
        resp = client.get(endpoint, params=query or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("ONS request failed for %s: %s", dataset, exc)
        return pd.DataFrame(columns=["Date", "value"])

    payload = resp.json()
    observations = []
    if isinstance(payload, dict):
        observations = payload.get("observations") or payload.get("items") or []
    records = []
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        date_value = _ons_extract_date(obs, time_dimension)
        if not date_value:
            continue
        value_obj = obs.get(value_field) if value_field else None
        if value_obj in (None, ""):
            value_obj = obs.get("observation")
        if value_obj in (None, ""):
            value_obj = obs.get("value")
        if isinstance(value_obj, dict):
            value_obj = value_obj.get("value") or value_obj.get("obs")
        if value_obj in (None, "", "."):
            continue
        try:
            numeric = float(str(value_obj).replace(",", ""))
        except (TypeError, ValueError):
            continue
        records.append({"Date": date_value, "value": numeric})

    df = _records_to_frame(records)
    if df.empty:
        return df
    if start:
        start_ts = pd.to_datetime(start, utc=True, errors="coerce")
        if pd.notna(start_ts):
            df = df[df["Date"] >= start_ts]
    if end:
        end_ts = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.notna(end_ts):
            df = df[df["Date"] <= end_ts]
    return df.reset_index(drop=True)


def _fetch_bank_of_england(
    spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop(
        "base_url", "https://www.bankofengland.co.uk/boeapps/iadb/v1/data"
    ).rstrip("/")
    dataset = params.pop("dataset", params.pop("dataflow", None))
    identifier = spec.identifier.strip("/")
    if not identifier and not dataset:
        return pd.DataFrame(columns=["Date", "value"])

    if dataset and "/" not in identifier:
        path = f"{dataset.strip('/')}/{identifier}" if identifier else dataset.strip("/")
    else:
        path = identifier or dataset.strip("/")

    if not path:
        return pd.DataFrame(columns=["Date", "value"])

    query: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
    if start:
        query.setdefault("startPeriod", start)
        query.setdefault("startTime", start)
    if end:
        query.setdefault("endPeriod", end)
        query.setdefault("endTime", end)
    query.setdefault("format", query.get("format", "sdmx-json"))

    endpoint = f"{base_url}/{path}"
    try:
        resp = client.get(endpoint, params=query or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("Bank of England request failed for %s: %s", path, exc)
        return pd.DataFrame(columns=["Date", "value"])

    records = list(_sdmx_collect_records(resp.json()))
    return _records_to_frame(records)


def _fetch_eurostat(
    spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop(
        "base_url", "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
    ).rstrip("/")
    identifier = spec.identifier.strip("/")
    if not identifier:
        return pd.DataFrame(columns=["Date", "value"])

    query: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
    if start:
        query.setdefault("startPeriod", start)
        query.setdefault("startTime", start)
    if end:
        query.setdefault("endPeriod", end)
        query.setdefault("endTime", end)
    query.setdefault("format", query.get("format", "SDMX-JSON"))

    endpoint = f"{base_url}/{identifier}"
    try:
        resp = client.get(endpoint, params=query or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("Eurostat request failed for %s: %s", identifier, exc)
        return pd.DataFrame(columns=["Date", "value"])

    records = list(_sdmx_collect_records(resp.json()))
    return _records_to_frame(records)


def _fetch_ecb(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop("base_url", "https://sdw-wsrest.ecb.europa.eu/service/data").rstrip("/")
    identifier = spec.identifier.strip("/")
    if not identifier:
        return pd.DataFrame(columns=["Date", "value"])

    query: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
    if start:
        query.setdefault("startPeriod", start)
        query.setdefault("startTime", start)
    if end:
        query.setdefault("endPeriod", end)
        query.setdefault("endTime", end)
    query.setdefault("format", query.get("format", "sdmx-json"))

    endpoint = f"{base_url}/{identifier}"
    try:
        resp = client.get(endpoint, params=query or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("ECB request failed for %s: %s", identifier, exc)
        return pd.DataFrame(columns=["Date", "value"])

    records = list(_sdmx_collect_records(resp.json()))
    return _records_to_frame(records)


def _fetch_bcb(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop(
        "base_url", "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
    ).rstrip("/")
    series = params.pop("series", params.pop("serie", None)) or spec.identifier
    series = str(series).strip("/")
    if not series:
        return pd.DataFrame(columns=["Date", "value"])

    endpoint = f"{base_url}/{series}/dados"
    query: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
    query.setdefault("formato", query.get("formato", "json"))
    if start:
        query.setdefault("dataInicial", start)
    if end:
        query.setdefault("dataFinal", end)

    try:
        resp = client.get(endpoint, params=query or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("Banco Central do Brasil request failed for %s: %s", series, exc)
        return pd.DataFrame(columns=["Date", "value"])

    payload = resp.json()
    if not isinstance(payload, list):
        return pd.DataFrame(columns=["Date", "value"])

    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        date_value = item.get("data") or item.get("date")
        value = item.get("valor") or item.get("value")
        if not date_value or value in (None, "", "."):
            continue
        date_text = str(date_value)
        if "/" in date_text and len(date_text.split("/")) == 3:
            day, month, year = date_text.split("/")
            date_text = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        try:
            numeric = float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            try:
                numeric = float(str(value))
            except (TypeError, ValueError):
                continue
        records.append({"Date": date_text, "value": numeric})

    return _records_to_frame(records)


def _oecd_time_labels(structure: Any) -> list[str]:
    if not isinstance(structure, dict):
        return []
    dimensions = structure.get("dimensions")
    if not isinstance(dimensions, dict):
        return []
    obs_dims = dimensions.get("observation")
    if not isinstance(obs_dims, list) or not obs_dims:
        return []
    time_dim = None
    for dim in obs_dims:
        if isinstance(dim, dict) and dim.get("id", "").upper() in {"TIME_PERIOD", "TIME"}:
            time_dim = dim
            break
    if time_dim is None:
        time_dim = obs_dims[-1] if isinstance(obs_dims[-1], dict) else None
    values = time_dim.get("values") if isinstance(time_dim, dict) else None
    if not isinstance(values, list):
        return []
    labels: list[str] = []
    for entry in values:
        if isinstance(entry, dict):
            label = entry.get("id") or entry.get("name") or entry.get("value")
        else:
            label = str(entry)
        if label is None:
            continue
        labels.append(str(label))
    return labels


def _oecd_extract_value(value: Any) -> Optional[float]:
    if isinstance(value, list):
        value = value[0] if value else None
    if value in (None, "", "."):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None


def _parse_oecd_observations(observations: Dict[str, Any], structure: Any) -> Iterable[Dict[str, object]]:
    time_labels = _oecd_time_labels(structure)
    records = []
    for key, raw in observations.items():
        if not isinstance(key, str):
            continue
        parts = key.split(":")
        try:
            time_idx = int(parts[-1])
        except (TypeError, ValueError):
            continue
        date = time_labels[time_idx] if time_idx < len(time_labels) else None
        value = _oecd_extract_value(raw)
        if date and value is not None:
            records.append({"Date": date, "value": value})
    return records


def _fetch_oecd(spec: SeriesSpec, client: "httpx.Client", start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    params = dict(spec.params)
    base_url = params.pop("base_url", "https://stats.oecd.org/SDMX-JSON/data").rstrip("/")
    identifier = spec.identifier.strip("/")
    if not identifier:
        return pd.DataFrame(columns=["Date", "value"])

    query: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
    if start:
        query.setdefault("startTime", start)
        query.setdefault("startPeriod", start)
    if end:
        query.setdefault("endTime", end)
        query.setdefault("endPeriod", end)

    endpoint = f"{base_url}/{identifier}"
    try:
        resp = client.get(endpoint, params=query or None, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failure is environment specific
        logger.warning("OECD request failed for %s: %s", identifier, exc)
        return pd.DataFrame(columns=["Date", "value"])

    payload = resp.json()
    data_sets = payload.get("dataSets") if isinstance(payload, dict) else None
    dataset = data_sets[0] if isinstance(data_sets, list) and data_sets else {}
    structure = payload.get("structure") if isinstance(payload, dict) else None

    records: Iterable[Dict[str, object]] = []
    if isinstance(dataset, dict):
        if isinstance(dataset.get("observations"), dict):
            records = _parse_oecd_observations(dataset["observations"], structure)
        elif isinstance(dataset.get("series"), dict):
            series_map = dataset["series"]
            for series in series_map.values():
                if isinstance(series, dict) and isinstance(series.get("observations"), dict):
                    records = _parse_oecd_observations(series["observations"], structure)
                    break
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
    "statcan": _fetch_statcan,
    "statistics_canada": _fetch_statcan,
    "bankofcanada": _fetch_bank_of_canada,
    "boc": _fetch_bank_of_canada,
    "open_canada": _fetch_open_canada,
    "opencanada": _fetch_open_canada,
    "ons": _fetch_ons,
    "office_for_national_statistics": _fetch_ons,
    "bankofengland": _fetch_bank_of_england,
    "bank_of_england": _fetch_bank_of_england,
    "boe": _fetch_bank_of_england,
    "eurostat": _fetch_eurostat,
    "ecb": _fetch_ecb,
    "european_central_bank": _fetch_ecb,
    "bcb": _fetch_bcb,
    "banco_central_do_brasil": _fetch_bcb,
    "banco_central_brasil": _fetch_bcb,
    "oecd": _fetch_oecd,
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

