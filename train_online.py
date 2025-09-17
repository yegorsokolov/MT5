import time
from pathlib import Path
import random
import logging
from typing import Any
import numpy as np
import pandas as pd
import joblib
import json
from river import compose, preprocessing, linear_model

from utils import load_config
from log_utils import setup_logging, log_exceptions
from data.live_recorder import load_ticks
from data.features import make_features
from data.labels import triple_barrier
from train_utils import resolve_training_features
from model_registry import save_model
from state_manager import watch_config

setup_logging()
logger = logging.getLogger(__name__)


@log_exceptions
def train_online(
    data_path: Path | str | None = None,
    model_dir: Path | str | None = None,
    *,
    min_ticks: int = 1000,
    interval: int = 300,
    run_once: bool = False,
) -> None:
    """Incrementally update a river model using recorded live ticks.

    Parameters
    ----------
    data_path: Path, optional
        Location of the recorded tick dataset. Defaults to ``data/live``.
    model_dir: Path, optional
        Directory where models are stored. Defaults to ``models``.
    min_ticks: int, optional
        Minimum number of new ticks required to trigger a training step.
    interval: int, optional
        Maximum number of seconds between training steps.
    run_once: bool, optional
        When ``True`` the function processes a single training step and
        returns immediately. Useful for tests.
    """

    cfg = load_config()
    _observer = watch_config(cfg)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    training_cfg = getattr(cfg, "training", None)

    def _cfg_value(name: str, default: Any) -> Any:
        if training_cfg is not None and hasattr(training_cfg, name):
            value = getattr(training_cfg, name)
        else:
            value = cfg.get(name, None)
        return default if value is None else value

    label_col = "tb_label"
    pt_mult = float(_cfg_value("pt_mult", 0.01))
    sl_mult = float(_cfg_value("sl_mult", 0.01))
    horizon_raw = int(_cfg_value("max_horizon", 5))
    label_horizon = max(1, horizon_raw)

    window_default = max(512, label_horizon * 2)
    try:
        feature_window = int(cfg.get("online_feature_window", window_default))
    except (TypeError, ValueError):
        feature_window = window_default
    feature_window = max(feature_window, label_horizon * 2)

    root = Path(__file__).resolve().parent
    data_path = Path(data_path) if data_path is not None else root / "data" / "live"
    model_dir = Path(model_dir) if model_dir is not None else root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    latest_path = model_dir / "online_latest.joblib"

    model = compose.Pipeline(
        preprocessing.StandardScaler(), linear_model.LogisticRegression()
    )

    def _ensure_timestamp(value: Any) -> pd.Timestamp | None:
        if value is None or isinstance(value, pd.Timestamp):
            return value
        try:
            return pd.Timestamp(value)
        except Exception:
            return None

    def _sanitize_timings(timings: Any) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        if not isinstance(timings, dict):
            return cleaned
        for name, entry in timings.items():
            if isinstance(entry, dict):
                cleaned[name] = {
                    key: float(val)
                    if isinstance(val, (int, float, np.floating))
                    else val
                    for key, val in entry.items()
                }
            elif isinstance(entry, (int, float, np.floating)):
                cleaned[name] = float(entry)
            else:
                cleaned[name] = entry
        return cleaned

    def _build_feature_metadata(
        columns: list[str] | None,
        timings: Any,
        last_tick: pd.Timestamp | None,
        last_train: pd.Timestamp | None,
        *,
        filled: list[str] | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "feature_columns": list(columns or []),
            "label_column": label_col,
            "feature_window": int(feature_window),
        }
        sanitized = _sanitize_timings(timings)
        if sanitized:
            meta["feature_timings"] = sanitized
        if last_tick is not None:
            meta["last_tick_ts"] = pd.Timestamp(last_tick).isoformat()
        if last_train is not None:
            meta["last_train_ts"] = pd.Timestamp(last_train).isoformat()
        if filled:
            meta["filled_missing_columns"] = list(filled)
        return meta

    last_tick_ts: pd.Timestamp | None = None
    last_train_ts: pd.Timestamp | None = None
    feature_columns: list[str] | None = None
    feature_metadata: dict[str, Any] = {}
    history = pd.DataFrame()

    if latest_path.exists():
        try:
            payload = joblib.load(latest_path)
            if isinstance(payload, dict):
                model = payload.get("model", model)
                last_tick_ts = _ensure_timestamp(payload.get("last_tick_ts"))
                last_train_ts = _ensure_timestamp(payload.get("last_train_ts"))
                stored_cols = payload.get("feature_columns")
                if stored_cols:
                    feature_columns = list(stored_cols)
                feature_metadata = payload.get("feature_metadata", {}) or {}
            else:
                model, last_tick_ts = payload
                last_train_ts = last_tick_ts
            logger.info("Loaded existing model from %s", latest_path)
        except Exception as exc:  # pragma: no cover - just warn
            logger.warning("Failed to load existing model: %s", exc)
            last_tick_ts = None
            last_train_ts = None
            feature_columns = None
            feature_metadata = {}

    last_train = time.time()
    new_ticks = 0
    start_ts = last_train_ts or last_tick_ts
    buffer_size = max(feature_window + label_horizon, label_horizon * 2)

    while True:
        df = load_ticks(data_path, last_tick_ts)
        if not df.empty:
            df = df.sort_values("Timestamp").reset_index(drop=True)
            history = pd.concat([history, df], ignore_index=True)
            history = history.sort_values("Timestamp").reset_index(drop=True)
            if len(history) > buffer_size:
                history = history.iloc[-buffer_size:].reset_index(drop=True)
            if not history.empty and "Timestamp" in history.columns:
                start_ts = history["Timestamp"].min()
                last_tick_ts = history["Timestamp"].max()

            base_frame = history.copy()
            if {"Bid", "Ask"}.issubset(base_frame.columns) and "mid" not in base_frame.columns:
                base_frame["mid"] = (base_frame["Bid"] + base_frame["Ask"]) / 2

            try:
                feature_frame = make_features(base_frame.copy(), validate=False)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Feature generation failed: %s", exc, exc_info=True)
                feature_frame = pd.DataFrame()

            if not feature_frame.empty:
                if "Timestamp" in base_frame.columns:
                    feature_frame["Timestamp"] = base_frame["Timestamp"]
                if "Symbol" in base_frame.columns and "Symbol" not in feature_frame.columns:
                    feature_frame["Symbol"] = base_frame["Symbol"]
                if "mid" not in feature_frame.columns and "mid" in base_frame.columns:
                    feature_frame["mid"] = base_frame["mid"]

                mid_series = feature_frame.get("mid")
                if mid_series is None:
                    logger.debug("Skipping update because 'mid' price is missing from features")
                else:
                    labels = triple_barrier(mid_series, pt_mult, sl_mult, label_horizon)
                    feature_frame[label_col] = labels
                    if label_horizon > 0 and len(feature_frame) > label_horizon:
                        train_frame = feature_frame.iloc[:-label_horizon].copy()
                    else:
                        train_frame = feature_frame.copy()

                    if last_train_ts is not None and "Timestamp" in train_frame.columns:
                        train_frame = train_frame[train_frame["Timestamp"] > last_train_ts]

                    train_frame = train_frame.replace([np.inf, -np.inf], np.nan)
                    train_frame.dropna(subset=[label_col], inplace=True)

                    if not train_frame.empty:
                        if feature_columns is None:
                            try:
                                training_section = getattr(cfg, "training", cfg)
                                feature_columns = resolve_training_features(
                                    train_frame,
                                    train_frame[label_col],
                                    training_section,
                                    id_columns={"Timestamp", "Symbol"},
                                    target_columns={label_col},
                                )
                            except Exception as exc:  # pragma: no cover - best effort
                                logger.warning(
                                    "Falling back to numeric feature set: %s",
                                    exc,
                                )
                                numeric_cols = train_frame.select_dtypes(
                                    include=[np.number, "bool"]
                                ).columns
                                skip_cols = {label_col, "Timestamp", "Symbol"}
                                feature_columns = [
                                    col
                                    for col in train_frame.columns
                                    if col in numeric_cols and col not in skip_cols
                                ]

                        updates = 0
                        filled_missing: list[str] = []
                        if feature_columns:
                            feature_columns = list(feature_columns)
                            for col in feature_columns:
                                if col not in train_frame.columns:
                                    train_frame[col] = 0.0
                                    filled_missing.append(col)

                            train_frame = train_frame.copy()
                            train_frame[feature_columns] = train_frame[feature_columns].fillna(0.0)

                            for _, row in train_frame.iterrows():
                                feature_values = row[feature_columns].to_dict()
                                feature_values = {
                                    key: float(val)
                                    if isinstance(val, (np.generic, float, int, bool))
                                    else val
                                    for key, val in feature_values.items()
                                }
                                y_val = 1 if row[label_col] > 0 else 0
                                model.learn_one(feature_values, y_val)
                                timestamp = row.get("Timestamp")
                                if timestamp is not None:
                                    last_train_ts = pd.Timestamp(timestamp)
                                updates += 1

                            if updates:
                                new_ticks += updates
                                feature_metadata = _build_feature_metadata(
                                    feature_columns,
                                    feature_frame.attrs.get("feature_timings"),
                                    last_tick_ts,
                                    last_train_ts,
                                    filled=filled_missing,
                                )

        now = time.time()
        if new_ticks >= min_ticks or (now - last_train) >= interval:
            if new_ticks > 0:
                state = {
                    "model": model,
                    "last_tick_ts": last_tick_ts,
                    "last_train_ts": last_train_ts,
                    "feature_columns": list(feature_columns or []),
                    "feature_metadata": feature_metadata,
                    "label_column": label_col,
                }
                joblib.dump(state, latest_path)
                version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                preprocessing_meta = dict(feature_metadata)
                if feature_columns is not None and "feature_columns" not in preprocessing_meta:
                    preprocessing_meta["feature_columns"] = list(feature_columns)
                preprocessing_meta.setdefault("label_column", label_col)
                preprocessing_meta.setdefault("feature_window", feature_window)
                if last_tick_ts is not None:
                    preprocessing_meta.setdefault("last_tick_ts", last_tick_ts.isoformat())
                if last_train_ts is not None:
                    preprocessing_meta.setdefault("last_train_ts", last_train_ts.isoformat())

                metadata = {
                    "drawdown_limit": cfg.get("drawdown_limit"),
                    "training_window": [
                        start_ts.isoformat() if start_ts else None,
                        last_train_ts.isoformat() if last_train_ts else None,
                    ],
                    "preprocessing": preprocessing_meta,
                }
                save_path = model_dir / f"online_{version}.pkl"
                save_model(
                    f"online_{version}",
                    model,
                    metadata,
                    save_path,
                )
                prov_log = model_dir / "training_provenance.log"
                try:
                    with prov_log.open("a", encoding="utf-8") as f:
                        record = {"version": version, **metadata}
                        f.write(json.dumps(record) + "\n")
                except Exception:  # pragma: no cover - best effort
                    logger.debug("Failed to write provenance to %s", prov_log)
                feature_count = len(preprocessing_meta.get("feature_columns", []))
                logger.info(
                    "Updated model %s with %d new ticks; window %s - %s drawdown_limit=%s feature_count=%d",
                    version,
                    new_ticks,
                    metadata["training_window"][0],
                    metadata["training_window"][1],
                    metadata["drawdown_limit"],
                    feature_count,
                )
                new_ticks = 0
                last_train = now
            if run_once:
                break
        if run_once:
            break
        time.sleep(1)


def rollback_model(
    version: str | None = None, model_dir: Path | str | None = None
) -> Path:
    """Restore a previously saved model version as the active one.

    Parameters
    ----------
    version:
        Timestamp identifier of the model to restore (e.g. ``20240101000000``).
        When omitted the function rolls back to the second most recent model.
    model_dir:
        Directory where model artifacts are stored. Defaults to ``models``.
    """

    root = Path(__file__).resolve().parent
    model_dir = Path(model_dir) if model_dir is not None else root / "models"
    versions = sorted(model_dir.glob("online_*.pkl"))
    if version:
        candidate = model_dir / f"online_{version}.pkl"
        if not candidate.exists():  # pragma: no cover - user input
            raise FileNotFoundError(candidate)
    else:
        if len(versions) < 2:
            raise ValueError("No previous model available for rollback")
        candidate = versions[-2]
    model = joblib.load(candidate)
    meta_path = candidate.with_suffix(".json")
    last_tick_ts: pd.Timestamp | None = None
    last_train_ts: pd.Timestamp | None = None
    feature_columns: list[str] = []
    feature_metadata: dict[str, Any] = {}
    label_col = "tb_label"
    if meta_path.exists():
        try:
            info = json.loads(meta_path.read_text())
            window = info.get("training_window", [None, None])
            if isinstance(window, list) and len(window) >= 2 and window[1]:
                last_train_ts = pd.Timestamp(window[1])
            preprocessing = info.get("preprocessing", {})
            if isinstance(preprocessing, dict):
                feature_metadata = dict(preprocessing)
                cols = preprocessing.get("feature_columns")
                if cols:
                    feature_columns = list(cols)
                label_col = preprocessing.get("label_column", label_col)
                tick_val = preprocessing.get("last_tick_ts")
                train_val = preprocessing.get("last_train_ts")
                if tick_val:
                    last_tick_ts = pd.Timestamp(tick_val)
                if train_val:
                    last_train_ts = pd.Timestamp(train_val)
        except Exception:  # pragma: no cover - best effort
            pass
    if last_tick_ts is None:
        last_tick_ts = last_train_ts
    latest_path = model_dir / "online_latest.joblib"
    state = {
        "model": model,
        "last_tick_ts": last_tick_ts,
        "last_train_ts": last_train_ts,
        "feature_columns": feature_columns,
        "feature_metadata": feature_metadata,
        "label_column": label_col,
    }
    joblib.dump(state, latest_path)
    logger.info("Rolled back to model %s", candidate.stem)
    return latest_path


if __name__ == "__main__":
    train_online()
