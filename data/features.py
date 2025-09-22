"""Lightweight feature engineering orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import hashlib
import json
import inspect
import os
import shutil
from functools import wraps

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Memory

try:  # pragma: no cover - optional dependency
    from data.polars_utils import to_polars_df, to_pandas_df
except Exception:  # pragma: no cover - polars optional
    to_polars_df = to_pandas_df = None  # type: ignore

from analytics.metrics_store import record_metric
from features import get_feature_pipeline
from features.news import (
    add_economic_calendar_features,
    add_news_sentiment_features,
)
from features.cross_asset import (
    add_index_features,
    add_cross_asset_features,
)
from analysis.cross_spectral import (
    compute as cross_spectral_compute,
    REQUIREMENTS as CROSS_SPECTRAL_REQ,
)
from analysis.dtw_features import add_dtw_features
from analysis.knowledge_graph import (
    load_knowledge_graph,
    risk_score,
    opportunity_score,
)
from analysis.regime_detection import periodic_reclassification
from utils.resource_monitor import monitor, ResourceCapabilities
from utils import load_config
from mt5.config_models import ConfigError
from analysis import feature_gate
from analysis.data_lineage import log_lineage
from analysis.fractal_features import rolling_fractal_features
from analysis.frequency_features import spectral_features, wavelet_energy, stl_decompose
from analysis.garch_vol import garch_volatility
from features.validators import validate_ge
from .multitimeframe import aggregate_timeframes
from feature_store import register_feature, load_feature
from analysis.feature_evolver import FeatureEvolver
from analysis.anomaly_detector import detect_anomalies
from risk.funding_costs import fetch_funding_info

logger = logging.getLogger(__name__)

try:
    LATENCY_THRESHOLD = float(load_config().features.latency_threshold)
except ConfigError:  # pragma: no cover - config may be unavailable
    LATENCY_THRESHOLD = 0.0

try:  # pragma: no cover - backend configuration optional
    BACKEND = load_config().features.backend  # type: ignore[attr-defined]
except Exception:
    BACKEND = "pandas"

# Location for persisted factor exposure matrices
FACTOR_EXPOSURE_DIR = Path("data/factors")

# Joblib cache for feature computations
_FEATURE_CACHE_DIR = Path(os.environ.get("FEATURE_CACHE_DIR", "data/feature_cache"))
_memory = Memory(str(_FEATURE_CACHE_DIR), verbose=0)


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def hash_dataframe(df: pd.DataFrame) -> str:
    """Return a SHA256 hash for the given DataFrame.

    The hash is computed from the pandas hash of the frame to ensure that
    equivalent frames produce identical hashes regardless of index ordering.
    """

    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def _version_hash(df: pd.DataFrame, feat_cfg: object) -> str:
    """Compute a deterministic hash for ``df`` and ``feat_cfg``."""
    m = hashlib.sha1()
    m.update(hash_dataframe(df).encode())
    m.update(json.dumps(feat_cfg, sort_keys=True).encode())
    return m.hexdigest()


def _get_code_signature(func) -> str:
    override = os.environ.get("FEATURE_CACHE_CODE_HASH")
    if override:
        return override
    try:
        file_path = Path(inspect.getfile(func))
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "0"


def _enforce_cache_limit():
    max_gb = os.environ.get("FEATURE_CACHE_MAX_GB")
    if not max_gb:
        return
    try:
        max_bytes = float(max_gb) * 1024**3
    except ValueError:
        logger.warning("Invalid FEATURE_CACHE_MAX_GB: %s", max_gb)
        return
    if not _FEATURE_CACHE_DIR.exists():
        return
    total = _dir_size(_FEATURE_CACHE_DIR)
    if total <= max_bytes:
        return
    entries = []
    for entry in _FEATURE_CACHE_DIR.iterdir():
        entry_size = _dir_size(entry)
        entry_atime = entry.stat().st_atime
        entries.append((entry_atime, entry_size, entry))
    entries.sort()
    for _, size, entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass
        total -= size
        if total <= max_bytes:
            break


def cache_feature(func):
    """Cache feature computation based on raw data, module name and code signature."""

    def _cached(cache_key, df, *args, **kwargs):
        return func(df, *args, **kwargs)

    cached_impl = _memory.cache(_cached, ignore=["df"])

    @wraps(func)
    def wrapper(df, *args, **kwargs):
        if os.environ.get("NO_CACHE"):
            return func(df, *args, **kwargs)
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()
        module = func.__module__
        code_sig = _get_code_signature(func)
        cache_key = hashlib.sha256(
            "|".join((data_hash, module, code_sig)).encode()
        ).hexdigest()
        result = cached_impl(cache_key, df, *args, **kwargs)
        _enforce_cache_limit()
        return result

    return wrapper


def _apply_transform(fn, df):
    """Apply ``fn`` to ``df`` unless throttled by latency."""
    latency = getattr(
        monitor, "latency", lambda: getattr(monitor, "tick_to_signal_latency", 0.0)
    )
    latency_val = latency() if callable(latency) else latency
    if (
        LATENCY_THRESHOLD
        and latency_val > LATENCY_THRESHOLD
        and getattr(fn, "degradable", False)
    ):
        logger.warning(
            "Throttling feature %s (latency %.3fs > %.3fs)",
            fn.__name__,
            latency_val,
            LATENCY_THRESHOLD,
        )
        try:
            record_metric("feature_throttled", 1.0, {"feature": fn.__name__})
        except Exception:
            pass
        return df
    if BACKEND == "polars" and to_polars_df and getattr(fn, "supports_polars", False):
        pl_df = to_polars_df(df)
        result = fn(pl_df)
        return to_pandas_df(result)
    return fn(df)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast common numeric dtypes to reduce memory usage.

    ``float64`` columns are converted to ``float32`` and ``int64`` columns
    to ``int32``.  The transformation is applied in-place and the original
    DataFrame is returned for convenience.
    """

    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings to ``df``.

    The following columns are appended based on the ``Timestamp`` column:

    - ``hour_of_day_sin`` / ``hour_of_day_cos``: Sine and cosine transforms
      of the hour within the day.
    - ``day_of_week_sin`` / ``day_of_week_cos``: Sine and cosine transforms of
      the day of week where Monday=0.
    """

    if "Timestamp" not in df.columns:
        return df

    times = pd.to_datetime(df["Timestamp"], utc=True)
    hours = times.dt.hour + times.dt.minute / 60.0
    df = df.copy()
    df["hour_of_day_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_of_day_cos"] = np.cos(2 * np.pi * hours / 24)
    dow = times.dt.dayofweek
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


@cache_feature
def add_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append spread, swap rate and leverage related features."""

    if "Symbol" not in df.columns:
        return df
    df = df.copy()
    if {"Bid", "Ask"}.issubset(df.columns):
        df["spread"] = df["Ask"] - df["Bid"]
    try:
        infos = {s: fetch_funding_info(s) for s in df["Symbol"].unique()}
        df["swap_rate"] = df["Symbol"].map(lambda s: infos[s].swap_rate)
        df["margin_requirement"] = df["Symbol"].map(
            lambda s: infos[s].margin_requirement
        )
        mr = df["margin_requirement"].replace(0, np.inf)
        df["account_leverage"] = 1.0 / mr
    except Exception:
        df["swap_rate"] = 0.0
        df["margin_requirement"] = 0.0
        df["account_leverage"] = 0.0
    return df


@cache_feature
def add_alt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge alternative datasets into ``df``.

    This function loads shipping metrics, retail transactions, macro
    indicators, news sentiment, Kalshi prediction market aggregates and
    weather series (as well as legacy options/on-chain/ESG metrics) using
    :func:`data.alt_data_loader.load_alt_data` and performs a
    ``merge_asof`` on ``Timestamp`` and ``Symbol``.  Missing columns are
    forward filled and remaining gaps replaced with ``0`` so downstream
    models can rely on their presence.
    """

    if "Symbol" not in df.columns:
        return df

    from .alt_data_loader import load_alt_data

    alt = load_alt_data(sorted(df["Symbol"].unique()))
    if not alt.empty:
        alt = alt.rename(columns={"Date": "alt_date"})
        df = pd.merge_asof(
            df.sort_values("Timestamp"),
            alt.sort_values("alt_date"),
            left_on="Timestamp",
            right_on="alt_date",
            by="Symbol",
            direction="backward",
        ).drop(columns=["alt_date"])

    for col in [
        "implied_vol",
        "active_addresses",
        "esg_score",
        "shipping_metric",
        "retail_sales",
        "temperature",
        "gdp",
        "cpi",
        "interest_rate",
        "news_sentiment",
        "kalshi_total_open_interest",
        "kalshi_total_daily_volume",
        "kalshi_total_block_volume",
        "kalshi_total_high",
        "kalshi_total_low",
        "kalshi_market_count",
        "kalshi_open_interest",
        "kalshi_daily_volume",
        "kalshi_block_volume",
        "kalshi_high",
        "kalshi_low",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    return df


# Mark as degradable for high-latency throttling
add_alt_features.degradable = True  # type: ignore[attr-defined]


@cache_feature
def add_factor_exposure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge latest factor exposures into ``df``.

    Exposure matrices are stored under ``data/factors`` with filenames of the
    form ``exposures_<timestamp>.csv`` produced by
    :mod:`analysis.factor_updater`.  The most recent file is loaded and merged
    on ``Symbol``.  Each exposure column (``factor_1``, ``factor_2``, ...)
    is appended to ``df`` with missing values filled with ``0`` so downstream
    models can rely on their presence.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a ``Symbol`` column.

    Returns
    -------
    pd.DataFrame
        Original ``df`` with factor exposure columns appended when available.
    """

    if "Symbol" not in df.columns:
        return df

    try:
        files = sorted(FACTOR_EXPOSURE_DIR.glob("exposures_*.csv"))
        if not files:
            return df
        latest = files[-1]
        exposures = pd.read_csv(latest, index_col=0)
    except Exception:  # pragma: no cover - optional file may be missing
        logger.debug("factor exposure load failed", exc_info=True)
        return df

    exposures = exposures.rename_axis("Symbol").reset_index()
    factor_cols = [c for c in exposures.columns if c != "Symbol"]
    df = df.merge(exposures, on="Symbol", how="left")
    for col in factor_cols:
        df[col] = df[col].fillna(0.0)
    return df


def add_knowledge_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append knowledge graph-derived risk and opportunity scores."""

    company_col = None
    if "Symbol" in df.columns:
        company_col = "Symbol"
    elif "company" in df.columns:
        company_col = "company"
    if company_col is None:
        return df

    try:
        g = load_knowledge_graph()
    except Exception:  # pragma: no cover - graph may be unavailable
        df = df.copy()
        df["graph_risk"] = 0.0
        df["graph_opportunity"] = 0.0
        return df

    companies = df[company_col].astype(str).unique()
    risk_map = {c: risk_score(g, c) for c in companies}
    opp_map = {c: opportunity_score(g, c) for c in companies}

    df = df.copy()
    df["graph_risk"] = df[company_col].map(risk_map).fillna(0.0)
    df["graph_opportunity"] = df[company_col].map(opp_map).fillna(0.0)
    return df


def add_fractal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append rolling fractal metrics to ``df``.

    The ``mid`` price series is passed to
    :func:`analysis.fractal_features.rolling_fractal_features`, producing two
    columns that characterise price path complexity:

    - ``hurst``: Rolling Hurst exponent indicating trend persistence.
    - ``fractal_dim``: Katz fractal dimension of the price trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a ``mid`` price column.

    Returns
    -------
    pd.DataFrame
        Original ``df`` with ``hurst`` and ``fractal_dim`` appended. If ``mid``
        is missing the frame is returned unchanged.
    """

    if "mid" not in df.columns:
        return df

    feats = rolling_fractal_features(df["mid"])
    df = df.copy()
    df["hurst"] = feats["hurst"]
    df["fractal_dim"] = feats["fractal_dim"]
    return df


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append frequency-domain energy metrics to ``df``.

    The price series (``mid`` if present otherwise ``close``) is passed to
    :func:`analysis.frequency_features.spectral_features` and
    :func:`analysis.frequency_features.wavelet_energy` producing two columns:

    - ``spec_energy``: Rolling spectral energy derived from the FFT.
    - ``wavelet_energy``: Energy of wavelet detail coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a price column.

    Returns
    -------
    pd.DataFrame
        Original ``df`` with the new frequency features appended. If neither a
        ``mid`` nor ``close`` column is present the frame is returned
        unchanged.
    """

    price_col = None
    if "mid" in df.columns:
        price_col = "mid"
    elif "close" in df.columns:
        price_col = "close"
    if price_col is None:
        return df

    spec = spectral_features(df[price_col])
    wave = wavelet_energy(df[price_col])
    df = df.copy()
    df["spec_energy"] = spec["spec_energy"]
    df["wavelet_energy"] = wave["wavelet_energy"]
    return df


def add_stl_features(df: pd.DataFrame, period: int = 24) -> pd.DataFrame:
    """Append STL seasonal and trend components to ``df``.

    The price series (``mid`` if present otherwise ``close``) is decomposed via
    :func:`analysis.frequency_features.stl_decompose`, producing two columns:

    - ``stl_seasonal``: Extracted seasonal component.
    - ``stl_trend``: Extracted trend component.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a price column.
    period : int, default 24
        Length of the seasonal cycle supplied to STL.

    Returns
    -------
    pd.DataFrame
        Original ``df`` with the new STL features appended. If neither a
        ``mid`` nor ``close`` column is present the frame is returned
        unchanged.
    """

    price_col = None
    if "mid" in df.columns:
        price_col = "mid"
    elif "close" in df.columns:
        price_col = "close"
    if price_col is None:
        return df

    comp = stl_decompose(df[price_col], period=period)
    df = df.copy()
    df["stl_seasonal"] = comp["seasonal"]
    df["stl_trend"] = comp["trend"]
    return df


def add_garch_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EGARCH volatility on returns and append ``garch_vol`` column.

    The heavy lifting is performed by :func:`analysis.garch_vol.garch_volatility`,
    which fits an EGARCH(1,1) model when the ``arch`` library is available and
    otherwise falls back to a 30-period rolling standard deviation. If the
    required ``return`` column is missing, the input frame is returned
    unchanged.
    """

    if "return" not in df.columns:
        return df

    df = df.copy()
    df["garch_vol"] = garch_volatility(df["return"])
    return df


def add_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Merge dividend, split and ownership data into ``df``.

    The function loads corporate action datasets via
    :func:`data.corporate_actions.load_corporate_actions` and performs a
    ``merge_asof`` on ``Timestamp`` and ``Symbol``.  Missing columns are
    forward filled with ``0`` to provide stable inputs for downstream
    models.  The loader is marked with ``min_capability='standard'`` so
    constrained environments can omit these heavier features.
    """

    if "Symbol" not in df.columns:
        return df

    from .corporate_actions import load_corporate_actions

    actions = load_corporate_actions(sorted(df["Symbol"].unique()))
    if not actions.empty:
        actions = actions.rename(columns={"Date": "corp_date"})
        df = pd.merge_asof(
            df.sort_values("Timestamp"),
            actions.sort_values("corp_date"),
            left_on="Timestamp",
            right_on="corp_date",
            by="Symbol",
            direction="backward",
        ).drop(columns=["corp_date"])

    for col in [
        "dividend",
        "split",
        "insider_trades",
        "institutional_holdings",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    return df


# Minimum capability required for corporate actions
add_corporate_actions.min_capability = "standard"  # type: ignore[attr-defined]
# Mark as degradable for latency-based throttling
add_corporate_actions.degradable = True  # type: ignore[attr-defined]


def add_cross_spectral_features(df: pd.DataFrame, window: int = 64) -> pd.DataFrame:
    """Append rolling cross-spectral coherence columns to ``df``.

    For each pair of symbols the magnitude-squared coherence of their return
    series is computed over ``window`` observations using
    :func:`analysis.cross_spectral.compute`.  For rows where ``Symbol`` equals
    ``A`` the column ``coh_B`` stores coherence with symbol ``B``.
    """

    caps = getattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0),
    )
    req = CROSS_SPECTRAL_REQ
    if (
        caps.cpus >= req.cpus
        and caps.memory_gb >= req.memory_gb
        and (caps.has_gpu or not req.has_gpu)
    ):
        try:
            cross_spectral_compute.degradable = True  # type: ignore[attr-defined]
            df = _apply_transform(
                lambda d: cross_spectral_compute(d, window=window), df
            )
        except Exception:  # pragma: no cover - computation may fail
            logger.debug("cross spectral computation failed", exc_info=True)
    return df


def make_features(df: pd.DataFrame, validate: bool = False) -> pd.DataFrame:
    """Generate model features by executing registered modules sequentially.

    In addition to the lightweight feature modules registered in
    :func:`features.get_feature_pipeline`, this function optionally merges
    external datasets such as fundamentals, options implied volatility and
    on-chain metrics.  These heavier data sources are only loaded when the
    :class:`utils.resource_monitor.ResourceMonitor` reports sufficient
    capabilities to avoid overwhelming constrained environments.  The final
    feature set is then passed through :func:`analysis.feature_gate.select` to
    drop low-importance or heavy features for the current capability tier and
    market regime, ensuring consistent behaviour across runs. After price-based
    features are computed, frequency-domain energies ``spec_energy`` and
    ``wavelet_energy`` along with rolling fractal metrics ``hurst`` and
    ``fractal_dim`` derived from the mid price are appended to the frame.  If
    factor exposure matrices have been generated by
    :mod:`analysis.factor_updater`, the latest exposures are merged via
    :func:`add_factor_exposure_features` prior to feature scaling.
    """

    try:
        cfg = load_config()
        feat_cfg = cfg.get("features", [])
    except Exception:  # pragma: no cover - config optional in tests
        feat_cfg = []
        cfg = {}

    version = _version_hash(df, feat_cfg)
    try:
        return load_feature(version)
    except FileNotFoundError:
        logger.debug("feature version %s not found in store", version)

    try:
        mtf = aggregate_timeframes(df, ["1min", "15min", "1h"]).drop(
            columns=["Timestamp"]
        )
        df = pd.concat([df, mtf], axis=1)
    except Exception:
        logger.debug("multi-timeframe aggregation failed", exc_info=True)

    df = add_time_features(df)
    df = add_cost_features(df)

    pipeline = list(get_feature_pipeline())
    cointegration_enabled = (
        isinstance(feat_cfg, dict) and feat_cfg.get("cointegration")
    ) or (isinstance(feat_cfg, list) and "cointegration" in feat_cfg)
    if cointegration_enabled:
        from features import cointegration as _cointegration

        if _cointegration.compute not in pipeline:
            pipeline.append(_cointegration.compute)

    import concurrent.futures
    import threading
    import time

    merge_lock = threading.Lock()
    feature_timings: dict[str, dict[str, float]] = {}

    def _record_duration(name: str, start: float, end: float) -> None:
        duration = end - start
        entry = feature_timings.setdefault(name, {})
        entry.update({"start": start, "end": end, "duration": duration})
        try:
            record_metric("feature_runtime_seconds", duration, {"feature": name})
        except Exception:
            pass

    def _record_columns(name: str, cols: Iterable[str]) -> None:
        entry = feature_timings.setdefault(name, {})
        entry["columns"] = len(cols)
        try:
            record_metric(
                "feature_columns_added", float(len(cols)), {"feature": name}
            )
        except Exception:
            pass

    def _apply_with_metrics(func, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        start = time.perf_counter()
        prev_cols = set(data.columns)
        result = func(data, *args, **kwargs)
        end = time.perf_counter()
        name = getattr(func, "__name__", "unknown") or "unknown"
        logger.info("feature %s executed in %.3fs", name, end - start)
        _record_duration(name, start, end)
        new_cols = [c for c in result.columns if c not in prev_cols]
        _record_columns(name, new_cols)
        run_id = result.attrs.get("run_id", "unknown")
        raw_file = result.attrs.get("source", "unknown")
        for col in new_cols:
            log_lineage(run_id, raw_file, name, col)
        return result

    def _run_compute(compute):
        start = time.perf_counter()
        result = compute(df.copy())
        end = time.perf_counter()
        return compute.__name__, result, start, end

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_run_compute, compute) for compute in pipeline]
        for future in concurrent.futures.as_completed(futures):
            name, res, start, end = future.result()
            duration = end - start
            logger.info("feature %s executed in %.3fs", name, duration)
            with merge_lock:
                results[name] = res
                _record_duration(name, start, end)

    for compute in pipeline:
        res = results[compute.__name__]
        prev_cols = set(df.columns)
        new_cols = [c for c in res.columns if c not in prev_cols]
        with merge_lock:
            df[new_cols] = res[new_cols]
        run_id = df.attrs.get("run_id", "unknown")
        raw_file = df.attrs.get("source", "unknown")
        for col in new_cols:
            log_lineage(run_id, raw_file, compute.__name__, col)
        _record_columns(compute.__name__, new_cols)

    # GARCH volatility derived from returns
    df = _apply_with_metrics(add_garch_volatility, df)

    # Cross-spectral coherence metrics between symbols
    df = _apply_with_metrics(add_cross_spectral_features, df)

    # Dynamic time warping distances between select symbol pairs
    df = _apply_with_metrics(add_dtw_features, df)

    # Knowledge graph-based risk and opportunity scores
    df = _apply_with_metrics(add_knowledge_graph_features, df)

    # Frequency-domain energy metrics on the price series
    df = _apply_with_metrics(add_frequency_features, df)

    # Seasonal-trend decomposition of the price series
    df = _apply_with_metrics(add_stl_features, df)

    # Fractal metrics derived from mid-price after price-based features
    df = _apply_with_metrics(add_fractal_features, df)

    # Estimate market and VAE regimes for gating features
    try:
        step_val = cfg.get("regime_reclass_period", 500)

        def _reclass(frame: pd.DataFrame) -> pd.DataFrame:
            return periodic_reclassification(frame, step=step_val)

        _reclass.__name__ = "periodic_reclassification"  # type: ignore[attr-defined]
        df = _apply_with_metrics(_reclass, df)
    except Exception:
        logger.debug("regime classification failed", exc_info=True)

    evolver = FeatureEvolver()
    df = _apply_with_metrics(evolver.apply_stored_features, df)
    feat_opts = cfg.get("features", {}) if isinstance(cfg.get("features"), dict) else {}
    target_col = feat_opts.get("target_col")
    if not target_col:
        for cand in ("target", "return", "y"):
            if cand in df.columns:
                target_col = cand
                break
    if target_col:
        def _hyper(frame: pd.DataFrame) -> pd.DataFrame:
            return evolver.apply_hypernet(frame, target_col=target_col)

        _hyper.__name__ = "apply_hypernet"  # type: ignore[attr-defined]
        df = _apply_with_metrics(_hyper, df)
    if feat_opts.get("auto_evolve") and target_col:
        module_path = Path("feature_store") / "evolved_features.py"

        def _maybe(frame: pd.DataFrame) -> pd.DataFrame:
            return evolver.maybe_evolve(
                frame,
                target_col=target_col,
                module_path=module_path,
            )

        _maybe.__name__ = "maybe_evolve"  # type: ignore[attr-defined]
        df = _apply_with_metrics(_maybe, df)

    # Allow runtime plugins to extend the feature set
    adjacency = df.attrs.get("adjacency_matrices")
    try:
        import dataset  # type: ignore

        plugins = getattr(dataset, "FEATURE_PLUGINS", [])
    except Exception:
        plugins = []
    for plugin in plugins:
        def _plugin_wrapper(frame: pd.DataFrame, _plugin=plugin) -> pd.DataFrame:
            return _plugin(frame, adjacency_matrices=adjacency)

        name = getattr(plugin, "__name__", "feature_plugin")
        _plugin_wrapper.__name__ = name  # type: ignore[attr-defined]
        df = _apply_with_metrics(_plugin_wrapper, df)

    # Merge options implied volatility and skew where resources allow
    try:
        from . import options_vol, vol_skew

        caps = getattr(
            monitor,
            "capabilities",
            ResourceCapabilities(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0),
        )
        modules = [
            (options_vol, ["implied_vol", "vol_skew"]),
            (vol_skew, ["vol_skew"]),
        ]
        for mod, cols in modules:
            req = getattr(mod, "REQUIREMENTS", caps)
            if (
                caps.cpus >= req.cpus
                and caps.memory_gb >= req.memory_gb
                and (caps.has_gpu or not req.has_gpu)
            ):
                mod.compute.degradable = True  # type: ignore[attr-defined]
                df = _apply_transform(mod.compute, df)
            else:
                for col in cols:
                    if col not in df.columns:
                        df[col] = 0.0
    except Exception:
        logger.debug("options volatility merge failed", exc_info=True)
        for col in ["implied_vol", "vol_skew"]:
            if col not in df.columns:
                df[col] = 0.0

    tier = getattr(monitor, "capability_tier", "lite")
    if tier in {"standard", "gpu", "hpc"}:
        try:
            from .fundamental_loader import load_fundamental_data

            if "Symbol" in df.columns:
                fundamentals = load_fundamental_data(sorted(df["Symbol"].unique()))
            else:
                fundamentals = pd.DataFrame()
            if not fundamentals.empty and "Symbol" in df.columns:
                fundamentals = fundamentals.rename(columns={"Date": "fund_date"})
                df = pd.merge_asof(
                    df.sort_values("Timestamp"),
                    fundamentals.sort_values("fund_date"),
                    left_on="Timestamp",
                    right_on="fund_date",
                    by="Symbol",
                    direction="backward",
                ).drop(columns=["fund_date"])
            for col in [
                "revenue",
                "net_income",
                "pe_ratio",
                "dividend_yield",
            ]:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].ffill().fillna(0.0)
        except Exception:  # pragma: no cover - optional dependency failures
            logger.debug("fundamental data merge failed", exc_info=True)
            for col in [
                "revenue",
                "net_income",
                "pe_ratio",
                "dividend_yield",
            ]:
                if col not in df.columns:
                    df[col] = 0.0
        try:
            from .macro_features import load_macro_features

            df = load_macro_features(df)
        except Exception:
            logger.debug("macro feature merge failed", exc_info=True)
            for col in ["macro_gdp", "macro_cpi", "macro_interest_rate"]:
                if col not in df.columns:
                    df[col] = 0.0
        try:
            from news.stock_headlines import load_scores as load_headline_scores

            if "Symbol" in df.columns:
                headlines = load_headline_scores()
            else:
                headlines = pd.DataFrame()
            if not headlines.empty and "Symbol" in df.columns:
                headlines = headlines.rename(
                    columns={"symbol": "Symbol", "timestamp": "headline_time"}
                )
                df = pd.merge_asof(
                    df.sort_values("Timestamp"),
                    headlines.sort_values("headline_time"),
                    left_on="Timestamp",
                    right_on="headline_time",
                    by="Symbol",
                    direction="backward",
                ).drop(columns=["headline_time"])
            if "news_movement_score" not in df.columns:
                df["news_movement_score"] = 0.0
        except Exception:
            logger.debug("news movement merge failed", exc_info=True)
            df["news_movement_score"] = 0.0
        try:
            from news.sentiment_score import load_vectors as load_news_vectors

            vectors = load_news_vectors()
            if not vectors.empty and "Symbol" in df.columns:
                vectors = vectors.rename(
                    columns={"symbol": "Symbol", "timestamp": "news_time"}
                )
                df = pd.merge_asof(
                    df.sort_values("Timestamp"),
                    vectors.sort_values("news_time"),
                    left_on="Timestamp",
                    right_on="news_time",
                    by="Symbol",
                    direction="backward",
                ).drop(columns=["news_time"])
            for k in range(3):
                if f"news_sentiment_{k}" not in df.columns:
                    df[f"news_sentiment_{k}"] = 0.0
                if f"news_impact_{k}" not in df.columns:
                    df[f"news_impact_{k}"] = 0.0
                if f"news_severity_{k}" not in df.columns:
                    df[f"news_severity_{k}"] = 0.0
                if f"news_sentiment_effect_{k}" not in df.columns:
                    df[f"news_sentiment_effect_{k}"] = 0.0
                if f"news_length_score_{k}" not in df.columns:
                    df[f"news_length_score_{k}"] = 0.0
                if f"news_effect_minutes_{k}" not in df.columns:
                    df[f"news_effect_minutes_{k}"] = 0.0
                if f"news_effect_half_life_{k}" not in df.columns:
                    df[f"news_effect_half_life_{k}"] = 0.0
            if "news_risk_scale" not in df.columns:
                df["news_risk_scale"] = 1.0
        except Exception:
            logger.debug("news sentiment merge failed", exc_info=True)
            for k in range(3):
                df[f"news_sentiment_{k}"] = 0.0
                df[f"news_impact_{k}"] = 0.0
                df[f"news_severity_{k}"] = 0.0
                df[f"news_sentiment_effect_{k}"] = 0.0
                df[f"news_length_score_{k}"] = 0.0
                df[f"news_effect_minutes_{k}"] = 0.0
                df[f"news_effect_half_life_{k}"] = 0.0
            df["news_risk_scale"] = 1.0
        try:
            df = _apply_transform(add_corporate_actions, df)
        except Exception:
            logger.debug("corporate actions merge failed", exc_info=True)
            for col in [
                "dividend",
                "split",
                "insider_trades",
                "institutional_holdings",
            ]:
                if col not in df.columns:
                    df[col] = 0.0
    if tier in {"gpu", "hpc"}:
        try:
            df = _apply_transform(add_alt_features, df)
        except Exception:
            logger.debug("alternative data merge failed", exc_info=True)
            for col in [
                "implied_vol",
                "active_addresses",
                "esg_score",
                "shipping_metric",
                "retail_sales",
                "temperature",
            ]:
                if col not in df.columns:
                    df[col] = 0.0

    if validate:
        try:
            from .validators import FEATURE_SCHEMA

            FEATURE_SCHEMA.validate(df, lazy=True)
        except Exception:
            logger.debug("feature validation failed", exc_info=True)

    # Drop features based on capability tier and regime specific gating.  The
    # gate is computed offline and persisted for deterministic behaviour, so at
    # runtime we simply apply the stored selection without recomputing
    # importances.
    regime_id = (
        int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 0
    )
    df, _ = feature_gate.select(df, tier, regime_id, persist=False)

    # Append latest factor exposures before scaling
    df = _apply_with_metrics(add_factor_exposure_features, df)
    start = time.perf_counter()
    prev_cols = set(df.columns)
    df, anomalies = detect_anomalies(df, method="isolation_forest")
    end = time.perf_counter()
    _record_duration("detect_anomalies", start, end)
    new_cols = [c for c in df.columns if c not in prev_cols]
    _record_columns("detect_anomalies", new_cols)
    if not anomalies.empty:
        rate = len(anomalies) / (len(df) + len(anomalies))
        logger.info("anomaly_rate=%.4f", rate)

    df = optimize_dtypes(df)
    df.attrs["feature_timings"] = feature_timings

    if validate:
        validate_ge(df, "engineered_features")
    register_feature(version, df, {"features": feat_cfg})
    return df


def make_features_memmap(path: str | Path, chunk_size: int = 1000) -> pd.DataFrame:
    """Load history from ``path`` and compute features."""
    df = pd.read_parquet(path)
    return make_features(df, validate=False)


# -- Technical helpers -------------------------------------------------


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def ma_cross_signal(
    df: pd.DataFrame, short: str = "ma_10", long: str = "ma_30"
) -> pd.Series:
    cross_up = (df[short] > df[long]) & (df[short].shift(1) <= df[long].shift(1))
    cross_down = (df[short] < df[long]) & (df[short].shift(1) >= df[long].shift(1))
    signal = pd.Series(0, index=df.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def train_test_split(
    df: pd.DataFrame, n_train: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "Symbol" in df.columns:
        order = df.groupby("Symbol").cumcount()
        train = df[order < n_train].copy()
        test = df[order >= n_train].copy()
        return train, test
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test


def make_sequence_arrays(
    df: pd.DataFrame, features: List[str], seq_len: int, label_col: str = "return"
) -> Tuple[np.ndarray, np.ndarray]:
    if label_col != "return" and label_col not in df.columns:
        raise KeyError(label_col)

    feature_count = len(features)
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    groups = [df]
    if "Symbol" in df.columns:
        groups = [g for _, g in df.groupby("Symbol")]

    for g in groups:
        values = g[features].to_numpy()
        if values.shape[0] <= seq_len:
            continue

        if label_col == "return":
            target_series = (g["return"].shift(-1) > 0).astype(np.int64).to_numpy()
            end = len(g) - 1
        else:
            target_series = g[label_col].to_numpy()
            end = len(g)

        window_count = end - seq_len
        if window_count <= 0:
            continue

        windows = sliding_window_view(values, window_shape=seq_len, axis=0)
        X_list.append(np.asarray(windows[:window_count]))
        y_list.append(np.asarray(target_series[seq_len:end]))

    if X_list:
        X = np.concatenate(X_list, axis=0)
    else:
        X = np.empty((0, seq_len, feature_count), dtype=float)

    if y_list:
        y = np.concatenate(y_list, axis=0)
    else:
        y_dtype = np.int64 if label_col == "return" else df[label_col].dtype
        y = np.empty((0,), dtype=y_dtype)
    return X, y


def apply_evolved_features(
    df: pd.DataFrame, store_dir: str | Path | None = None
) -> pd.DataFrame:
    """Apply evolved features stored in ``feature_store`` to ``df``."""

    from analysis.feature_evolver import FeatureEvolver

    evolver = FeatureEvolver(store_dir=store_dir)
    return evolver.apply_stored_features(df)


__all__ = [
    "add_index_features",
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "add_cross_asset_features",
    "add_cross_spectral_features",
    "add_dtw_features",
    "add_frequency_features",
    "add_stl_features",
    "add_fractal_features",
    "add_garch_volatility",
    "add_factor_exposure_features",
    "add_knowledge_graph_features",
    "add_time_features",
    "add_cost_features",
    "make_features",
    "make_features_memmap",
    "compute_rsi",
    "ma_cross_signal",
    "train_test_split",
    "make_sequence_arrays",
    "apply_evolved_features",
]

# Evolved features will be appended below by ``FeatureEvolver``.
