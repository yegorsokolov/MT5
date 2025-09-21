from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "config"


def load_config_fixture(name: str) -> dict:
    path = FIXTURE_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict)
    return dict(data)


def test_update_config_preserves_secret_placeholder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_data = load_config_fixture("update_config.yaml")
    cfg_file.write_text(yaml.safe_dump(cfg_data))

    log_path = tmp_path / "logs" / "config_changes.csv"
    monkeypatch.setattr(utils, "_LOG_PATH", log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("MLFLOW_PASS", "resolved-secret")

    utils.update_config("plugin_cache_ttl", 120, "Adjust TTL")

    saved_text = cfg_file.read_text()
    assert "resolved-secret" not in saved_text

    saved = yaml.safe_load(saved_text)
    assert saved["mlflow"]["password"] == "secret://MLFLOW_PASS"
    assert saved["plugin_cache_ttl"] == 120

    assert cfg_data["plugin_cache_ttl"] == 60

    assert log_path.exists()
    log_lines = log_path.read_text().strip().splitlines()
    assert any("plugin_cache_ttl" in line for line in log_lines)
    assert any("Adjust TTL" in line for line in log_lines)
