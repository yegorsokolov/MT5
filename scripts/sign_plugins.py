#!/usr/bin/env python3
from __future__ import annotations

"""Sign plugin modules using the repository's signing key.

This utility is intended for maintainers to generate ``.sig`` files for new or
updated plugins.  Signatures are written as base64 encoded text to keep the
repository free of binary blobs.  The private key used for signing is not
stored in the repository and must be provided explicitly.
"""

import argparse
from pathlib import Path

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "plugin_security", REPO_ROOT / "utils" / "plugin_security.py"
)
plugin_security = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(plugin_security)  # type: ignore
sign_plugin = plugin_security.sign_plugin


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign plugin modules")
    parser.add_argument("plugins", nargs="+", help="Paths to plugin modules to sign")
    parser.add_argument(
        "--key", required=True, help="Path to the private key in PEM format"
    )
    args = parser.parse_args()

    key_path = Path(args.key)
    for mod in args.plugins:
        path = Path(mod)
        sign_plugin(path, key_path)
        print(f"Signed {path}")


if __name__ == "__main__":
    main()
