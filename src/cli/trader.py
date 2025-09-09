"""Simple trading CLI interface."""
from __future__ import annotations

import argparse

from utils.config import get_account_settings
from broker.client import BrokerClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trading CLI")
    parser.add_argument(
        "--account",
        choices=["demo", "live"],
        help="Override account environment (demo or live)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_account_settings()
    if args.account:
        settings["environment"] = args.account
    client = BrokerClient(settings)
    client.connect()


if __name__ == "__main__":  # pragma: no cover
    main()
