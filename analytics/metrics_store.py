from __future__ import annotations

"""Persistence layer for daily performance metrics.

This module stores daily aggregates such as return, Sharpe ratio and
maximum drawdown.  Metrics are persisted to a Parquet file which can be
loaded as a time–series for analysis or reporting.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import pandas as pd
import pandas.api.types as ptypes
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.types as patypes
from filelock import FileLock

DEFAULT_PATH = Path("analytics/metrics.parquet")
# Separate store for generic time-series metrics
TS_PATH = Path("analytics/metrics_timeseries.parquet")

FieldDef = Tuple[str, pa.DataType]

METRICS_BASE_FIELDS: Sequence[FieldDef] = (
    ("date", pa.timestamp("ns")),
    ("return", pa.float64()),
    ("sharpe", pa.float64()),
    ("drawdown", pa.float64()),
    ("regime", pa.string()),
    ("__ingested_at", pa.timestamp("ns", tz="UTC")),
)

TS_BASE_FIELDS: Sequence[FieldDef] = (
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("name", pa.string()),
    ("value", pa.float64()),
)

TYPE_REGISTRY: Dict[str, pa.DataType] = {
    "double": pa.float64(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "string": pa.string(),
    "bool": pa.bool_(),
    "timestamp[ns]": pa.timestamp("ns"),
    "timestamp[ns, tz=UTC]": pa.timestamp("ns", tz="UTC"),
}


def _lock_path(path: Path) -> Path:
    """Return the lock file path for a dataset target."""

    return path.parent / f"{path.name}.lock"


def _schema_file(path: Path) -> Path:
    """Return the metadata file storing the dataset schema."""

    return Path(path) / "_schema.json"


def _schema_from_fields(fields: Sequence[FieldDef]) -> pa.Schema:
    """Construct a ``pyarrow`` schema from ``fields`` preserving order."""

    unique: Dict[str, pa.Field] = {}
    order: list[str] = []
    for name, dtype in fields:
        if name not in unique:
            unique[name] = pa.field(name, dtype, nullable=True)
            order.append(name)
    return pa.schema([unique[name] for name in order])


def _inspect_dataset_schema(path: Path) -> Optional[pa.Schema]:
    """Return the schema for an existing parquet dataset if available."""

    path = Path(path)
    if path.is_file():
        try:
            return pq.read_schema(path)
        except (FileNotFoundError, OSError):
            return None
    if path.is_dir():
        for file in sorted(path.glob("*.parquet")):
            try:
                return pq.read_schema(file)
            except (FileNotFoundError, OSError):
                continue
    return None


def _load_schema(path: Path, base_fields: Sequence[FieldDef]) -> pa.Schema:
    """Load the persisted schema for ``path`` if available."""

    schema_path = _schema_file(path)
    if schema_path.exists():
        try:
            payload = json.loads(schema_path.read_text())
            fields_payload = payload.get("fields", [])
            if isinstance(fields_payload, list):
                base_names = {name for name, _ in base_fields}
                base_mapping = {name: dtype for name, dtype in base_fields}
                loaded: list[FieldDef] = []
                seen: set[str] = set()
                for item in fields_payload:
                    name = item.get("name") if isinstance(item, dict) else None
                    type_name = item.get("type") if isinstance(item, dict) else None
                    if not isinstance(name, str) or name in seen:
                        continue
                    dtype = TYPE_REGISTRY.get(type_name) if isinstance(type_name, str) else None
                    if name in base_mapping:
                        dtype = base_mapping[name]
                    if dtype is None:
                        dtype = TYPE_REGISTRY.get("string", pa.string())
                    loaded.append((name, dtype))
                    seen.add(name)
                for name, dtype in base_fields:
                    if name not in seen:
                        loaded.append((name, dtype))
                if loaded:
                    return _schema_from_fields(loaded)
        except json.JSONDecodeError:
            pass

    discovered = _inspect_dataset_schema(path)
    if discovered is not None:
        base_names = {name for name, _ in base_fields}
        base_mapping = {name: dtype for name, dtype in base_fields}
        combined: list[FieldDef] = list(base_fields)
        for field in discovered:
            if field.name in base_names:
                continue
            dtype = TYPE_REGISTRY.get(str(field.type), pa.string())
            combined.append((field.name, dtype))
        return _schema_from_fields(combined)

    return _schema_from_fields(base_fields)


def _store_schema(path: Path, schema: pa.Schema) -> None:
    """Persist the ordered schema definition for ``path``."""

    schema_path = _schema_file(path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
            }
            for field in schema
        ]
    }
    schema_path.write_text(json.dumps(payload))


def _infer_arrow_type(value: Any) -> pa.DataType:
    """Infer an Arrow data type for ``value`` used in tag columns."""

    if value is None or value is pd.NA:
        return pa.string()
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int) and not isinstance(value, bool):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, (pd.Timestamp,)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            return pa.timestamp("ns", tz="UTC")
        return pa.timestamp("ns")
    return pa.string()


def _infer_arrow_type_from_series(series: pd.Series) -> pa.DataType:
    """Infer an Arrow type for an existing pandas series."""

    non_null = series.dropna()
    if non_null.empty:
        return pa.string()
    if ptypes.is_bool_dtype(non_null):
        return pa.bool_()
    if ptypes.is_integer_dtype(non_null):
        return pa.int64()
    if ptypes.is_float_dtype(non_null):
        return pa.float64()
    if ptypes.is_datetime64tz_dtype(non_null):
        tzinfo = getattr(non_null.dt.tz, "zone", None) or str(non_null.dt.tz)
        return pa.timestamp("ns", tz=str(tzinfo))
    if ptypes.is_datetime64_dtype(non_null):
        return pa.timestamp("ns")
    return _infer_arrow_type(non_null.iloc[0])


def _schema_with_field(schema: pa.Schema, name: str, dtype: pa.DataType) -> pa.Schema:
    """Return ``schema`` with ``name`` ensured to exist with ``dtype``."""

    fields: list[pa.Field] = []
    replaced = False
    for field in schema:
        if field.name == name:
            if field.type == dtype:
                return schema
            fields.append(pa.field(name, dtype, nullable=True))
            replaced = True
        else:
            fields.append(field.with_nullable(True))
    if not replaced:
        fields.append(pa.field(name, dtype, nullable=True))
    return pa.schema(fields)


def _coerce_scalar(value: Any, dtype: pa.DataType) -> Any:
    """Coerce ``value`` to a Python scalar compatible with ``dtype``."""

    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if patypes.is_boolean(dtype):
        return bool(value)
    if patypes.is_integer(dtype):
        return int(value)
    if patypes.is_floating(dtype):
        return float(value)
    if patypes.is_string(dtype):
        return str(value)
    if patypes.is_timestamp(dtype):
        ts = pd.Timestamp(value)
        tz = dtype.tz  # type: ignore[attr-defined]
        if tz:
            if ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            else:
                ts = ts.tz_convert(tz)
        else:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()
    return value


def _coerce_dataframe_to_schema(df: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    """Return a dataframe aligned with ``schema`` values and column order."""

    aligned = df.copy()
    for field in schema:
        if field.name not in aligned.columns:
            aligned[field.name] = None
        aligned[field.name] = aligned[field.name].apply(lambda v: _coerce_scalar(v, field.type))
    return aligned[list(schema.names)]


def _append_dataset_row(path: Path, row: Dict[str, Any], schema: pa.Schema) -> None:
    """Append ``row`` to the parquet dataset located at ``path``."""

    arrays = []
    for field in schema:
        value = _coerce_scalar(row.get(field.name), field.type)
        arrays.append(pa.array([value], type=field.type))
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_to_dataset(table, root_path=path, schema=schema)


def _prepare_metrics_legacy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise historical metrics data prior to dataset migration."""

    frame = df.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    else:
        frame["date"] = pd.NaT
    if "__ingested_at" in frame.columns:
        frame["__ingested_at"] = pd.to_datetime(frame["__ingested_at"], utc=True)
    else:
        frame["__ingested_at"] = pd.Timestamp("1970-01-01", tz="UTC")
    if "regime" in frame.columns:
        frame["regime"] = frame["regime"].astype("string")
    return frame


def _prepare_timeseries_legacy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise historical time-series metric rows."""

    frame = df.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    else:
        frame["timestamp"] = pd.NaT
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _ensure_dataset(
    path: Path,
    base_fields: Sequence[FieldDef],
    *,
    prepare_legacy: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pa.Schema:
    """Ensure ``path`` represents a parquet dataset directory and return its schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.is_file():
        existing = pd.read_parquet(path).reset_index(drop=True)
        if prepare_legacy is not None:
            existing = prepare_legacy(existing)
        base_names = {name for name, _ in base_fields}
        schema_fields: list[FieldDef] = list(base_fields)
        for column in existing.columns:
            if column not in base_names:
                inferred = _infer_arrow_type_from_series(existing[column])
                schema_fields.append((column, inferred))
        schema = _schema_from_fields(schema_fields)
        legacy_path = path.with_suffix(path.suffix + ".legacy")
        if legacy_path.exists():
            legacy_path.unlink()
        path.rename(legacy_path)
        path.mkdir(parents=True, exist_ok=True)
        if not existing.empty:
            prepared = _coerce_dataframe_to_schema(existing, schema)
            table = pa.Table.from_pandas(prepared, schema=schema, preserve_index=False, safe=False)
            pq.write_to_dataset(table, root_path=path, schema=schema)
        _store_schema(path, schema)
        legacy_path.unlink(missing_ok=True)
        return schema

    if path.is_dir():
        schema = _load_schema(path, base_fields)
        _store_schema(path, schema)
        return schema

    path.mkdir(parents=True, exist_ok=True)
    schema = _schema_from_fields(base_fields)
    _store_schema(path, schema)
    return schema


def _restore_regime_value(value: Any) -> Any:
    """Convert stored regime strings back to their original type."""

    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value
@dataclass
class MetricsStore:
    """Store and retrieve daily performance metrics.

    Parameters
    ----------
    path:
        Location of the Parquet file.  The directory is created if it does
        not already exist.
    """

    path: Path = DEFAULT_PATH

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        path = Path(self.path)

        if not path.exists():
            return pd.DataFrame(columns=["return", "sharpe", "drawdown", "regime"])

        schema = _load_schema(path, METRICS_BASE_FIELDS)
        names = list(schema.names)

        if path.is_file():
            try:
                df = pd.read_parquet(path)
            except (FileNotFoundError, OSError):
                return pd.DataFrame(columns=["return", "sharpe", "drawdown", "regime"])
        else:
            parquet_files = list(path.glob("*.parquet"))
            if not parquet_files:
                return pd.DataFrame(columns=["return", "sharpe", "drawdown", "regime"])
            dataset = ds.dataset(path, schema=schema)
            table = dataset.to_table()
            df = table.to_pandas()

        if df.empty:
            return pd.DataFrame(columns=["return", "sharpe", "drawdown", "regime"])

        for field in schema:
            if field.name not in df.columns:
                df[field.name] = pd.NA
        df = df[names]

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["__ingested_at"] = pd.to_datetime(df["__ingested_at"], utc=True)
        df["return"] = pd.to_numeric(df["return"], errors="coerce")
        df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")
        df["drawdown"] = pd.to_numeric(df["drawdown"], errors="coerce")

        df.sort_values(["date", "__ingested_at"], inplace=True)
        df = df.drop_duplicates(subset="date", keep="last")
        df.set_index("date", inplace=True)
        df["regime"] = df["regime"].apply(_restore_regime_value)
        return df[["return", "sharpe", "drawdown", "regime"]]

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Return all persisted metrics as a dataframe."""
        return self._load().copy()

    # ------------------------------------------------------------------
    def append(
        self,
        date: pd.Timestamp,
        *,
        ret: float,
        sharpe: float,
        drawdown: float,
        regime: Optional[int | str] = None,
    ) -> None:
        """Append a new day's metrics and persist to disk."""

        normalized_date = pd.Timestamp(date).normalize()
        row: Dict[str, Any] = {
            "date": normalized_date,
            "return": float(ret),
            "sharpe": float(sharpe),
            "drawdown": float(drawdown),
            "regime": None if regime is None else str(regime),
            "__ingested_at": pd.Timestamp.now(tz="UTC"),
        }

        dataset_path = Path(self.path)
        lock_file = _lock_path(dataset_path)
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(str(lock_file)):
            schema = _ensure_dataset(
                dataset_path,
                METRICS_BASE_FIELDS,
                prepare_legacy=_prepare_metrics_legacy_frame,
            )
            _append_dataset_row(dataset_path, row, schema)

    # ------------------------------------------------------------------
    def get(self, start: Optional[str | pd.Timestamp] = None, end: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        """Return metrics for a given date range."""

        df = self._load()
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df.copy()


# ---------------------------------------------------------------------------
def _ts_load(path: Path = TS_PATH) -> pd.DataFrame:
    """Load the time-series metrics dataframe."""

    path = Path(path)

    schema = _load_schema(path, TS_BASE_FIELDS)
    names = list(schema.names)
    if not path.exists():
        return pd.DataFrame(columns=names)

    if path.is_file():
        try:
            df = pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            return pd.DataFrame(columns=names)
    else:
        parquet_files = list(path.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame(columns=names)
        dataset = ds.dataset(path, schema=schema)
        table = dataset.to_table()
        df = table.to_pandas()

    if df.empty:
        return pd.DataFrame(columns=names)

    extra_columns = [col for col in df.columns if col not in schema.names]
    if extra_columns:
        for column in extra_columns:
            inferred = _infer_arrow_type_from_series(df[column])
            schema = _schema_with_field(schema, column, inferred)
        _store_schema(path, schema)
        names = list(schema.names)

    for field in schema:
        if field.name not in df.columns:
            df[field.name] = pd.NA

    df = df[names]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def record_metric(name: str, value: float, tags: Optional[Dict[str, Any]] = None, *, path: Path = TS_PATH) -> None:
    """Persist a single metric observation.

    Parameters
    ----------
    name:
        Metric name.
    value:
        Metric value.  Stored as ``float``.
    tags:
        Optional dictionary of additional columns to persist.
    path:
        Destination Parquet file.  Defaults to ``TS_PATH``.
    """

    path = Path(path)
    tags = tags or {}
    row: Dict[str, Any] = {
        "timestamp": pd.Timestamp.now(tz="UTC"),
        "name": name,
        "value": float(value),
    }
    row.update(tags)

    lock_file = _lock_path(path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_file)):
        schema = _ensure_dataset(path, TS_BASE_FIELDS, prepare_legacy=_prepare_timeseries_legacy_frame)

        schema_changed = False
        for column, value in row.items():
            if column not in schema.names:
                inferred = _infer_arrow_type(value)
                schema = _schema_with_field(schema, column, inferred)
                schema_changed = True

        _append_dataset_row(path, row, schema)

        if schema_changed:
            _store_schema(path, schema)


def model_cache_hit() -> None:
    """Record a cache hit for a loaded model."""
    record_metric("model_cache_hits", 1.0)


def model_unload() -> None:
    """Record that a cached model was unloaded."""
    record_metric("model_unloads", 1.0)


def query_metrics(
    name: str | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tags: Optional[Dict[str, Any]] = None,
    path: Path = TS_PATH,
) -> pd.DataFrame:
    """Return metrics from the time-series store.

    Filters by name, time range and optional tags.  The returned dataframe
    always includes ``timestamp``, ``name`` and ``value`` columns.
    """

    df = _ts_load(path)
    if name:
        df = df[df["name"] == name]
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if tags:
        for k, v in tags.items():
            if k in df.columns:
                df = df[df[k] == v]
            else:
                # No matching tag column implies no rows
                df = df.iloc[0:0]
                break
    return df.reset_index(drop=True)


def log_retrain_outcome(model: str, status: str) -> None:
    """Record the outcome of a retraining run.

    Parameters
    ----------
    model:
        Identifier of the model being retrained (e.g. ``"classic"``, ``"nn"`` or ``"rl"``).
    status:
        Outcome string such as ``"success"`` or ``"failed"``.
    """

    record_metric(
        "retrain_outcome",
        1.0 if status == "success" else 0.0,
        {"model": model, "status": status},
    )
