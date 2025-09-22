import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types


class _ValidationError(Exception):
    pass


def _identity_decorator(*args, **kwargs):
    def _decorator(func):
        return func

    return _decorator


def _field(default=None, **kwargs):
    return default


pydantic_stub = types.SimpleNamespace(
    BaseModel=object,
    ConfigDict=dict,
    Field=_field,
    field_validator=_identity_decorator,
    model_validator=_identity_decorator,
    ValidationError=_ValidationError,
)

sys.modules["pydantic"] = pydantic_stub
sys.modules["yaml"] = types.SimpleNamespace(
    safe_load=lambda *args, **kwargs: {},
    safe_dump=lambda *args, **kwargs: None,
)
mt5_config_stub = types.SimpleNamespace(AutoUpdateConfig=object)
mt5_package = types.ModuleType("mt5")
mt5_package.config_models = mt5_config_stub
mt5_package.__path__ = []
sys.modules["mt5"] = mt5_package
sys.modules["mt5.config_models"] = mt5_config_stub
utils_package = types.ModuleType("utils")
utils_package.PROJECT_ROOT = Path(__file__).resolve().parents[1]
utils_package.load_config = lambda: SimpleNamespace(auto_update=None, strategy=SimpleNamespace(symbols=[]))
sys.modules["utils"] = utils_package
market_hours_stub = types.ModuleType("utils.market_hours")
market_hours_stub.is_market_open = lambda exchange: False
utils_package.market_hours = market_hours_stub
sys.modules["utils.market_hours"] = market_hours_stub

from types import SimpleNamespace

from services.auto_updater import AutoUpdater


def _settings(**overrides):
    defaults = {
        "enabled": True,
        "remote": "origin",
        "branch": "main",
        "service_name": "mt5bot",
        "restart_command": None,
        "prefer_quiet_hours": True,
        "max_open_fraction": 0.5,
        "max_defer_minutes": 240,
        "fallback_exchange": "24/5",
        "exchanges": {},
        "protected_paths": ["logs", "reports", "checkpoints", "models"],
        "state_file": None,
        "lock_file": None,
        "dry_run": False,
        "check_interval_minutes": 15,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _completed(cmd, stdout="", returncode=0):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout)


def test_skip_when_repository_up_to_date(tmp_path, monkeypatch):
    settings = _settings(
        state_file=str(tmp_path / "state.json"),
        lock_file=str(tmp_path / "lock"),
    )
    updater = AutoUpdater(settings=settings, repo_path=tmp_path, symbols=["XAUUSD"], now_fn=lambda: datetime.now(UTC))

    calls = []

    expected = [
        (["git", "fetch", "origin", "main"], ""),
        (["git", "rev-parse", "origin/main"], "abc123\n"),
        (["git", "rev-parse", "HEAD"], "abc123\n"),
    ]

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        expected_cmd, stdout = expected.pop(0)
        assert cmd == expected_cmd
        return _completed(cmd, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    changed = updater.run()
    assert not changed
    assert [cmd for cmd in calls if cmd[:2] == ["git", "pull"]] == []
    assert not (tmp_path / "state.json").exists()


def test_defers_when_markets_open(tmp_path, monkeypatch):
    now = datetime(2024, 1, 1, tzinfo=UTC)
    settings = _settings(
        max_open_fraction=0.2,
        max_defer_minutes=180,
        exchanges={"XAUUSD": "24/5", "GBPUSD": "24/5"},
        state_file=str(tmp_path / "state.json"),
        lock_file=str(tmp_path / "lock"),
    )
    updater = AutoUpdater(settings=settings, repo_path=tmp_path, symbols=["XAUUSD", "GBPUSD"], now_fn=lambda: now)

    expected = [
        (["git", "fetch", "origin", "main"], ""),
        (["git", "rev-parse", "origin/main"], "def456\n"),
        (["git", "rev-parse", "HEAD"], "abc123\n"),
    ]

    def fake_run(cmd, **kwargs):
        expected_cmd, stdout = expected.pop(0)
        assert cmd == expected_cmd
        return _completed(cmd, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("services.auto_updater.is_market_open", lambda exchange: True)

    changed = updater.run()
    assert not changed
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["remote_commit"] == "def456"
    assert state["first_seen"].endswith("Z")
    assert state["last_checked"].endswith("Z")


def test_forces_update_after_max_deferral(tmp_path, monkeypatch):
    now = datetime(2024, 1, 1, tzinfo=UTC)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "remote_commit": "def456",
                "first_seen": (now - timedelta(hours=2)).isoformat(),
                "last_checked": (now - timedelta(hours=2)).isoformat(),
            }
        )
    )
    settings = _settings(
        max_open_fraction=0.0,
        max_defer_minutes=30,
        exchanges={"XAUUSD": "24/5"},
        state_file=str(state_path),
        lock_file=str(tmp_path / "lock"),
    )
    updater = AutoUpdater(settings=settings, repo_path=tmp_path, symbols=["XAUUSD"], now_fn=lambda: now)

    expected = [
        (["git", "fetch", "origin", "main"], ""),
        (["git", "rev-parse", "origin/main"], "def456\n"),
        (["git", "rev-parse", "HEAD"], "abc123\n"),
        (["git", "status", "--porcelain"], ""),
        (["git", "pull", "--ff-only", "origin", "main"], ""),
        (["systemctl", "restart", "mt5bot"], ""),
    ]

    def fake_run(cmd, **kwargs):
        expected_cmd, stdout = expected.pop(0)
        assert cmd == expected_cmd
        return _completed(cmd, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("services.auto_updater.is_market_open", lambda exchange: True)

    changed = updater.run()
    assert changed
    assert not state_path.exists()
    for folder in ["logs", "reports", "checkpoints", "models"]:
        assert (tmp_path / folder).is_dir()
    assert expected == []
