"""Tests for realtime startup helpers."""

import os
import builtins
import importlib
import itertools
import sys
from pathlib import Path
from typing import List

import pytest

pytest.importorskip("numpy", reason="realtime_train depends on numpy")
pytest.importorskip("pandas", reason="realtime_train depends on pandas")


os.environ.setdefault("SKIP_USER_RISK_PROMPT", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
realtime_train = importlib.import_module("realtime_train")


def _patch_non_interactive(monkeypatch) -> None:
    monkeypatch.setattr(realtime_train, "_stdin_supports_interaction", lambda: False)


def test_wait_for_terminal_login_returns_immediately(monkeypatch):
    """The helper should not prompt when already connected."""

    _patch_non_interactive(monkeypatch)
    calls: List[None] = []

    def _logged_in() -> bool:
        calls.append(None)
        return True

    monkeypatch.setattr(realtime_train.mt5_direct, "is_terminal_logged_in", _logged_in)
    monkeypatch.setattr(builtins, "input", pytest.fail)
    realtime_train._wait_for_terminal_login(timeout=None, max_retries=None)
    assert len(calls) == 1


def test_wait_for_terminal_login_retries_polling(monkeypatch):
    """When the terminal is offline the helper should retry with backoff."""

    _patch_non_interactive(monkeypatch)
    states = iter([False, False, True])
    calls: List[None] = []

    def _logged_in() -> bool:
        calls.append(None)
        return next(states)

    sleeps: List[float] = []

    monkeypatch.setattr(realtime_train.mt5_direct, "is_terminal_logged_in", _logged_in)
    monkeypatch.setattr(realtime_train.time, "sleep", sleeps.append)
    realtime_train._wait_for_terminal_login(timeout=None, max_retries=None)
    assert len(calls) == 3
    assert sleeps == [5.0, 7.5]


def test_wait_for_terminal_login_timeout(monkeypatch):
    """A timeout should raise a RuntimeError."""

    _patch_non_interactive(monkeypatch)
    monkeypatch.setattr(realtime_train.mt5_direct, "is_terminal_logged_in", lambda: False)
    counter = itertools.count()
    monkeypatch.setattr(realtime_train.time, "monotonic", lambda: next(counter))
    monkeypatch.setattr(realtime_train.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Timed out"):
        realtime_train._wait_for_terminal_login(timeout=2, max_retries=None)


def test_wait_for_terminal_login_max_retries(monkeypatch):
    """Exhausting retries should produce a clear error message."""

    _patch_non_interactive(monkeypatch)
    monkeypatch.setattr(realtime_train.mt5_direct, "is_terminal_logged_in", lambda: False)
    monkeypatch.setattr(realtime_train.time, "sleep", lambda _: None)
    msg = "MetaTrader 5 terminal login not detected after 4 attempts"
    with pytest.raises(RuntimeError, match=msg):
        realtime_train._wait_for_terminal_login(timeout=None, max_retries=3)


def test_wait_for_terminal_login_interactive_fallback(monkeypatch):
    """EOF on stdin should fall back to polling without blocking forever."""

    monkeypatch.setattr(realtime_train, "_stdin_supports_interaction", lambda: True)
    states = iter([False, False])

    def _logged_in() -> bool:
        try:
            return next(states)
        except StopIteration:
            return False

    sleeps: List[float] = []

    monkeypatch.setattr(realtime_train.mt5_direct, "is_terminal_logged_in", _logged_in)

    def _input(_: str) -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", _input)
    monkeypatch.setattr(realtime_train.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError):
        realtime_train._wait_for_terminal_login(timeout=None, max_retries=1)
    assert sleeps  # fallback triggered polling

