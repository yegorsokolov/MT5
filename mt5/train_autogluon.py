"""Backward compatible wrapper for the deprecated AutoGluon trainer."""

from __future__ import annotations

import warnings

from .train_tabular import main

warnings.warn(
    "mt5.train_autogluon has been replaced by mt5.train_tabular. "
    "Update your workflows to call 'python -m mt5.train_tabular' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["main"]
