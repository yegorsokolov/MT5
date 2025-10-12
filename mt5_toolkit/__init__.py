"""Compatibility wrapper exposing the MT5 CLI under a non-conflicting name."""

from mt5.__main__ import main

__all__ = ["main"]
