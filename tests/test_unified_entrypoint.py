from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt5 import __main__ as mt5_main


def _clear_mode_env(monkeypatch):
    monkeypatch.delenv("MT5_MODE", raising=False)
    monkeypatch.delenv("MT5_DEFAULT_MODE", raising=False)


def _run(argv: list[str]) -> int:
    return mt5_main.main(argv)


def test_default_mode_is_pipeline(monkeypatch, capsys):
    _clear_mode_env(monkeypatch)
    exit_code = _run(["--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "pipeline: mt5.pipeline_runner"


def test_environment_override_with_alias(monkeypatch, capsys):
    monkeypatch.setenv("MT5_MODE", "ReAl-TiMe")
    exit_code = _run(["--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "realtime: mt5.realtime_train"


def test_invalid_environment_falls_back_to_pipeline(monkeypatch, capsys):
    monkeypatch.setenv("MT5_MODE", "unknown")
    exit_code = _run(["--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "pipeline: mt5.pipeline_runner"


def test_config_override(monkeypatch, capsys):
    _clear_mode_env(monkeypatch)

    dummy_config = {"runtime": {"mode": "realtime"}}
    monkeypatch.setattr(mt5_main, "_load_config", lambda: dummy_config)

    exit_code = _run(["--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "realtime: mt5.realtime_train"


def test_cli_positional_overrides(monkeypatch, capsys):
    _clear_mode_env(monkeypatch)
    dummy_config = types.SimpleNamespace(get=lambda key, default=None: "train")
    monkeypatch.setattr(mt5_main, "_load_config", lambda: dummy_config)

    exit_code = _run(["backtest", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "backtest: mt5.backtest"


def test_cli_option_overrides(monkeypatch, capsys):
    _clear_mode_env(monkeypatch)
    exit_code = _run(["--mode", "backtest", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "backtest: mt5.backtest"


def test_list_modes(capsys):
    exit_code = _run(["--list"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip().splitlines()
    parsed = {
        name.strip(): rest.strip()
        for name, rest in (line.split("->", maxsplit=1) for line in out)
    }
    expected = {
        name: f"{entry.module} ({entry.description})"
        for name, entry in sorted(mt5_main.ENTRY_POINTS.items())
    }
    assert parsed == expected


def test_unknown_mode_errors(monkeypatch, capsys):
    _clear_mode_env(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        mt5_main.main(["--mode", "does-not-exist"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_missing_module_guard(monkeypatch, capsys):
    _clear_mode_env(monkeypatch)
    original_find_spec = mt5_main.importlib.util.find_spec

    def fake_find_spec(module: str, *args, **kwargs):
        if module == "mt5.realtime_train":
            return None
        return original_find_spec(module, *args, **kwargs)

    monkeypatch.setattr(mt5_main.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(SystemExit) as exc:
        mt5_main.main(["realtime"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "mt5.realtime_train" in err


def test_remainder_arguments_are_forwarded(monkeypatch):
    _clear_mode_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_run(module: str, argv):
        captured["module"] = module
        captured["argv"] = list(argv)

    monkeypatch.setattr(mt5_main, "_run_module", fake_run)
    exit_code = mt5_main.main(["backtest", "--foo", "bar", "--baz"])
    assert exit_code == 0
    assert captured == {
        "module": "mt5.backtest",
        "argv": ["--foo", "bar", "--baz"],
    }

