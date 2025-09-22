from __future__ import annotations

import argparse
from pprint import pprint

from feature_store import list_versions, purge_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect feature store versions")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List stored feature versions")
    purge = sub.add_parser("purge", help="Remove a stored version")
    purge.add_argument("version", help="Version hash to purge")

    args = parser.parse_args()

    if args.cmd == "list":
        pprint(list_versions())
    elif args.cmd == "purge":
        purge_version(args.version)


if __name__ == "__main__":
    main()
