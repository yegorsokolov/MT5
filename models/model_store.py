from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import shutil
import joblib

# Extra metadata helpers ------------------------------------------------------
def _save_metadata(name: str, data: Dict, store_dir: Path | None = None) -> str:
    """Persist a small JSON blob under the model store.

    This is used for recording auxiliary information such as replay statistics
    or tuned hyper-parameters.  The function creates a unique file name to avoid
    collisions and returns the resulting identifier which mirrors the behaviour
    of :func:`save_model`.
    """

    store = _ensure_store(store_dir)
    version_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    path = store / f"{name}_{version_id}.json"
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path.name

STORE_DIR = Path(__file__).resolve().parent / "store"


def _ensure_store(store_dir: Path | None = None) -> Path:
    store = store_dir or STORE_DIR
    store.mkdir(parents=True, exist_ok=True)
    return store


def save_model(
    model: Any | str | Path,
    training_config: Dict,
    performance: Dict,
    architecture_history: List[Dict] | None = None,
    store_dir: Path | None = None,
    features: List[str] | None = None,
) -> str:
    """Persist a model artifact with associated metadata.

    Parameters
    ----------
    model:
        The model object or path to an existing artifact.
    training_config:
        Configuration dictionary used for training.
    performance:
        Performance metrics dictionary.
    store_dir:
        Optional custom store directory for testing.

    Returns
    -------
    str
        Generated version identifier.
    """
    store = _ensure_store(store_dir)
    version_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    version_dir = store / version_id
    version_dir.mkdir(parents=True)

    artifact_name = "model.joblib"
    if isinstance(model, (str, Path)):
        src = Path(model)
        artifact_name = src.name
        dest = version_dir / artifact_name
        if src.is_dir():
            shutil.copytree(src, dest)
            hash_obj = hashlib.sha256()
            for file in sorted(dest.rglob("*")):
                if file.is_file():
                    hash_obj.update(file.read_bytes())
            artifact_hash = hash_obj.hexdigest()
        else:
            shutil.copy2(src, dest)
            artifact_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
    else:
        dest = version_dir / artifact_name
        joblib.dump(model, dest)
        artifact_hash = hashlib.sha256(dest.read_bytes()).hexdigest()

    metadata = {
        "hash": artifact_hash,
        "training_config": training_config,
        "performance": performance,
        "timestamp": datetime.utcnow().isoformat(),
        "artifact": artifact_name,
    }
    if architecture_history:
        metadata["architecture_history"] = architecture_history
    if features is not None:
        metadata["features"] = features
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    return version_id


def load_model(version_id: str, store_dir: Path | None = None) -> Tuple[Any, Dict]:
    """Load a model and its metadata by version identifier."""
    store = _ensure_store(store_dir)
    version_dir = store / version_id
    meta_file = version_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Version {version_id} not found")
    with open(meta_file) as f:
        metadata = json.load(f)
    artifact_path = version_dir / metadata.get("artifact", "model.joblib")
    if artifact_path.suffix == ".joblib":
        model = joblib.load(artifact_path)
    else:
        model = artifact_path
    return model, metadata


def list_versions(store_dir: Path | None = None) -> List[Dict]:
    """Return metadata for all saved model versions."""
    store = _ensure_store(store_dir)
    versions: List[Dict] = []
    for d in sorted(store.iterdir()):
        meta = d / "metadata.json"
        if meta.exists():
            with open(meta) as f:
                data = json.load(f)
            data["version_id"] = d.name
            versions.append(data)
    return versions


# Convenience wrappers --------------------------------------------------------
def save_replay_stats(stats: Dict, store_dir: Path | None = None) -> str:
    """Persist replay statistics for later analysis."""

    return _save_metadata("replay", stats, store_dir)


def save_tuned_params(params: Dict, store_dir: Path | None = None) -> str:
    """Persist tuned hyper-parameters."""

    return _save_metadata("tuned", params, store_dir)


__all__ = [
    "save_model",
    "load_model",
    "list_versions",
    "save_replay_stats",
    "save_tuned_params",
]
