"""Generate strategy training samples.

This module provides utilities to create simple strategy examples for
training or evaluation.  Examples follow the Evaluate-Plan-Act (EPA)
structure defined in :func:`training.prompts.strategy_templates.epa_template`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

from training.config import StrategyConfig
from training.prompts.strategy_templates import epa_template


def generate_strategy_examples(
    template_fn: Callable[[str, str, StrategyConfig | None], Dict[str, str]],
    n_examples: int,
    config: StrategyConfig | None = None,
) -> List[Dict[str, str]]:
    """Generate ``n_examples`` strategy prompts using ``template_fn``.

    Parameters
    ----------
    template_fn:
        A callable that accepts ``goal`` and ``context`` strings and returns a
        mapping with ``evaluate``, ``plan`` and ``act`` instructions.
    n_examples:
        Number of examples to generate.
    config:
        Optional :class:`~training.config.StrategyConfig` to influence prompt
        generation.

    Returns
    -------
    list of dict
        Each element contains the ``goal``, ``context`` and the ``evaluate``,
        ``plan`` and ``act`` fields produced by ``template_fn``.
    """

    examples: List[Dict[str, str]] = []
    for i in range(n_examples):
        goal = f"Example goal {i}"
        context = f"Example context {i}"
        sample = template_fn(goal, context, config)
        sample["goal"] = goal
        sample["context"] = context
        examples.append(sample)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EPA strategy examples")
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/strategies/epa_examples.jsonl"),
        help="Output JSONL file",
    )
    args = parser.parse_args()

    examples = generate_strategy_examples(epa_template, args.n_examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()
