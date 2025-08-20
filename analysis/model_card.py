from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from data.versioning import compute_hash


def _hash_datasets(paths: Sequence[Path | str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for p in paths:
        path = Path(p)
        try:
            hashes[str(path)] = compute_hash(path)
        except Exception:
            hashes[str(path)] = "missing"
    return hashes


def generate(
    config: Dict[str, Any],
    dataset_paths: Sequence[Path | str],
    features: Iterable[str],
    metrics: Dict[str, Any],
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Create a model card in Markdown and JSON format.

    Parameters
    ----------
    config: Dict[str, Any]
        Training configuration dictionary.
    dataset_paths: Sequence[Path | str]
        Paths of datasets used for training. Hashes are computed for
        reproducibility.
    features: Iterable[str]
        List of feature names used for training.
    metrics: Dict[str, Any]
        Validation metrics gathered during training.
    output_dir: Path | str
        Directory where the model card files should be written.

    Returns
    -------
    tuple[Path, Path]
        Paths to the generated Markdown and JSON files respectively.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    dataset_hashes = _hash_datasets(dataset_paths)

    card: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "training_config": config,
        "dataset_hashes": dataset_hashes,
        "features": list(features),
        "metrics": metrics,
    }

    json_path = out_dir / f"model_card_{timestamp}.json"
    md_path = out_dir / f"model_card_{timestamp}.md"

    with json_path.open("w") as f:
        json.dump(card, f, indent=2, sort_keys=True)

    md_lines = [
        "# Model Card",
        f"Generated: {card['generated_at']}",
        "\n## Training Configuration\n",
        "```json",
        json.dumps(card["training_config"], indent=2, sort_keys=True),
        "```",
        "\n## Dataset Hashes\n",
    ]
    for path, digest in dataset_hashes.items():
        md_lines.append(f"- `{path}`: `{digest}`")
    md_lines.append("\n## Features\n")
    for feat in card["features"]:
        md_lines.append(f"- {feat}")
    md_lines.extend([
        "\n## Validation Metrics\n",
        "```json",
        json.dumps(card["metrics"], indent=2, sort_keys=True),
        "```",
    ])

    md_path.write_text("\n".join(md_lines))
    return md_path, json_path


__all__ = ["generate"]
