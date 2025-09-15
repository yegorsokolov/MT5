import time
from pathlib import Path
import random
import logging
import numpy as np
import pandas as pd
import joblib
import json
from river import compose, preprocessing, linear_model

from utils import load_config
from log_utils import setup_logging, log_exceptions
from data.live_recorder import load_ticks
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

    root = Path(__file__).resolve().parent
    data_path = Path(data_path) if data_path is not None else root / "data" / "live"
    model_dir = Path(model_dir) if model_dir is not None else root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    latest_path = model_dir / "online_latest.joblib"

    model = compose.Pipeline(
        preprocessing.StandardScaler(), linear_model.LogisticRegression()
    )
    last_ts = None
    if latest_path.exists():
        try:
            model, last_ts = joblib.load(latest_path)
            logger.info("Loaded existing model from %s", latest_path)
        except Exception as exc:  # pragma: no cover - just warn
            logger.warning("Failed to load existing model: %s", exc)
            last_ts = None

    last_train = time.time()
    new_ticks = 0
    start_ts = last_ts
    while True:
        df = load_ticks(data_path, last_ts)
        if not df.empty:
            start_ts = start_ts or df["Timestamp"].min()
            df = df.sort_values("Timestamp")
            df["mid"] = (df["Bid"] + df["Ask"]) / 2
            df["return"] = df["mid"].pct_change()
            df["next_ret"] = df["return"].shift(-1)
            df.dropna(subset=["next_ret"], inplace=True)
            for _, row in df.iterrows():
                x = {"return": row["return"]}
                y = int(row["next_ret"] > 0)
                model.learn_one(x, y)
                last_ts = row["Timestamp"]
            new_ticks += len(df)

        now = time.time()
        if new_ticks >= min_ticks or (now - last_train) >= interval:
            if new_ticks > 0:
                joblib.dump((model, last_ts), latest_path)
                version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                metadata = {
                    "drawdown_limit": cfg.get("drawdown_limit"),
                    "training_window": [
                        start_ts.isoformat() if start_ts else None,
                        last_ts.isoformat() if last_ts else None,
                    ],
                }
                save_path = model_dir / f"online_{version}.pkl"
                save_model(
                    f"online_{version}",
                    model,
                    metadata,
                    save_path,
                )
                # Persist provenance information for later auditability
                prov_log = model_dir / "training_provenance.log"
                try:
                    with prov_log.open("a", encoding="utf-8") as f:
                        record = {"version": version, **metadata}
                        f.write(json.dumps(record) + "\n")
                except Exception:  # pragma: no cover - best effort
                    logger.debug("Failed to write provenance to %s", prov_log)
                logger.info(
                    "Updated model %s with %d new ticks; window %s - %s drawdown_limit=%s",
                    version,
                    new_ticks,
                    metadata["training_window"][0],
                    metadata["training_window"][1],
                    metadata["drawdown_limit"],
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
    last_ts = None
    if meta_path.exists():
        try:
            info = json.loads(meta_path.read_text())
            end = info.get("training_window", [None, None])[1]
            if end:
                last_ts = pd.Timestamp(end)
        except Exception:  # pragma: no cover - best effort
            pass
    latest_path = model_dir / "online_latest.joblib"
    joblib.dump((model, last_ts), latest_path)
    logger.info("Rolled back to model %s", candidate.stem)
    return latest_path


if __name__ == "__main__":
    train_online()
