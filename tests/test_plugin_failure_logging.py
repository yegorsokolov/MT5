import builtins
import logging
import sys
import types
from pathlib import Path


def test_plugin_logs_warning_on_failure(monkeypatch, caplog):
    """Plugins that fail to import should emit a warning."""
    log_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _setup_logging(*args, **kwargs):
        log_calls.append((args, kwargs))
        return logging.getLogger()

    log_utils_stub = types.SimpleNamespace(setup_logging=_setup_logging)
    monkeypatch.setitem(sys.modules, "log_utils", log_utils_stub)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plugins.atr":
            raise RuntimeError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    sys.modules.pop("plugins", None)
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    with caplog.at_level(logging.WARNING):
        import plugins  # noqa: F401

    assert not log_calls
    assert isinstance(plugins.get_logger(), logging.Logger)
    assert any("atr" in r.getMessage() for r in caplog.records)
    sys.modules.pop("plugins", None)
