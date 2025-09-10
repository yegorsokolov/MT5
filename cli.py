import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from utils import load_config


def _prepare_config(
    config: Optional[Path],
    seed: Optional[int],
    steps: Optional[int],
    steps_key: Optional[str],
    n_jobs: Optional[int] = None,
    num_threads: Optional[int] = None,
    validate: Optional[bool] = None,
) -> Optional[str]:
    """Load config and apply overrides, returning temp file path if needed."""
    if config is not None:
        os.environ["CONFIG_FILE"] = str(config)
    cfg = load_config()
    modified = False
    if seed is not None:
        cfg["seed"] = seed
        modified = True
    if steps is not None:
        key = steps_key or "steps"
        cfg[key] = steps
        modified = True
    if n_jobs is not None:
        cfg["n_jobs"] = n_jobs
        modified = True
    if num_threads is not None:
        cfg["num_threads"] = num_threads
        modified = True
    if validate is not None:
        cfg["validate"] = validate
        modified = True
    if modified:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(cfg, tmp)
        tmp.close()
        os.environ["CONFIG_FILE"] = tmp.name
        return tmp.name
    return None


def train_cmd(args: argparse.Namespace) -> None:
    from train import main as train_main

    tmp = _prepare_config(
        args.config,
        args.seed,
        None,
        None,
        n_jobs=args.n_jobs,
        num_threads=None,
        validate=args.validate,
    )
    try:
        train_main()
    finally:
        if tmp:
            os.unlink(tmp)


def train_nn_cmd(args: argparse.Namespace) -> None:
    from train_nn import main as train_nn_main

    tmp = _prepare_config(
        args.config,
        args.seed,
        None,
        None,
        n_jobs=None,
        num_threads=args.num_threads,
        validate=args.validate,
    )
    try:
        train_nn_main()
    finally:
        if tmp:
            os.unlink(tmp)


def train_rl_cmd(args: argparse.Namespace) -> None:
    from train_rl import main as train_rl_main

    tmp = _prepare_config(
        args.config,
        args.seed,
        args.steps,
        "rl_steps",
        validate=args.validate,
    )
    try:
        train_rl_main()
    finally:
        if tmp:
            os.unlink(tmp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mt5")
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable feature caching"
    )
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Run classic training pipeline")
    p_train.add_argument("--seed", type=int, default=None, help="Random seed")
    p_train.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    p_train.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for LightGBM/scikit-learn",
    )
    p_train.add_argument(
        "--validate",
        action="store_true",
        help="Enable strict data validation",
    )
    p_train.set_defaults(func=train_cmd)

    p_nn = sub.add_parser("train-nn", help="Train neural network model")
    p_nn.add_argument("--seed", type=int, default=None, help="Random seed")
    p_nn.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    p_nn.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of CPU threads for PyTorch",
    )
    p_nn.add_argument(
        "--validate",
        action="store_true",
        help="Enable strict data validation",
    )
    p_nn.set_defaults(func=train_nn_cmd)

    p_rl = sub.add_parser("train-rl", help="Train reinforcement learning model")
    p_rl.add_argument("--seed", type=int, default=None, help="Random seed")
    p_rl.add_argument("--steps", type=int, default=None, help="Number of training steps")
    p_rl.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    p_rl.add_argument(
        "--validate",
        action="store_true",
        help="Enable strict data validation",
    )
    p_rl.set_defaults(func=train_rl_cmd)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "no_cache", False):
        os.environ["NO_CACHE"] = "1"
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    args.func(args)
    return 0


def train_entry() -> None:
    main(["train", *sys.argv[1:]])


def train_nn_entry() -> None:
    main(["train-nn", *sys.argv[1:]])


def train_rl_entry() -> None:
    main(["train-rl", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

