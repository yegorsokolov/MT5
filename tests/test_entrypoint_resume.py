from __future__ import annotations

import importlib


def _reload_main():
    module = importlib.import_module("mt5.__main__")
    return importlib.reload(module)


def test_main_adds_resume_flag(monkeypatch, tmp_path):
    module = _reload_main()
    monkeypatch.setenv("MT5_MODE", "train")

    captured: dict[str, list[str] | str] = {}

    def fake_run(module_name: str, argv):
        captured["module"] = module_name
        captured["argv"] = list(argv)

    monkeypatch.setattr(module, "_run_module", fake_run)
    monkeypatch.setattr(module, "_load_config", lambda: {"checkpoint_dir": str(tmp_path)})
    monkeypatch.setattr(module, "_load_latest_checkpoint", lambda *_: (3, {}))

    exit_code = module.main([])

    assert exit_code == 0
    assert captured["module"] == "mt5.train"
    assert "--resume-online" in captured["argv"]


def test_main_skips_resume_when_no_checkpoint(monkeypatch, tmp_path):
    module = _reload_main()
    monkeypatch.setenv("MT5_MODE", "train")

    calls: list[tuple[str, list[str]]] = []

    def fake_run(module_name: str, argv):
        calls.append((module_name, list(argv)))

    monkeypatch.setattr(module, "_run_module", fake_run)
    monkeypatch.setattr(module, "_load_config", lambda: {"checkpoint_dir": str(tmp_path)})
    monkeypatch.setattr(module, "_load_latest_checkpoint", lambda *_: None)

    module.main([])

    assert calls[0][0] == "mt5.train"
    assert "--resume-online" not in calls[0][1]
