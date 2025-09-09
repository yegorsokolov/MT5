import argparse
from typing import Any, Dict, Iterable, Optional

from src.modes import Mode
from src.strategy.executor import StrategyExecutor


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trading system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in Mode],
        default=Mode.TRAINING.value,
        help="Operational mode of the system",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None, strategy: Optional[Dict[str, Any]] = None) -> StrategyExecutor:
    """Entry point for the trading system.

    Returns the created :class:`StrategyExecutor` for further interaction in tests.
    """

    args = parse_args(argv)
    mode = Mode(args.mode)
    executor = StrategyExecutor(mode=mode, strategy=strategy or {})
    return executor


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
