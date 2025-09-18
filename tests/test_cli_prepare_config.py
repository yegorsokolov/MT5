import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

import cli
from tests.yaml_helpers import ensure_real_yaml


def test_prepare_config_overrides_written_to_temp_file(tmp_path, monkeypatch):
    ensure_real_yaml()

    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 1\n")
    base_data = {
        "seed": 1,
        "n_jobs": 2,
        "training": {"seed": 1},
    }

    class DummyConfig:
        def model_dump(self):
            return copy.deepcopy(base_data)

    monkeypatch.setattr(cli, "load_config", lambda: DummyConfig())

    previous = os.environ.get("CONFIG_FILE")
    if previous is None:
        monkeypatch.delenv("CONFIG_FILE", raising=False)
    else:
        monkeypatch.setenv("CONFIG_FILE", previous)

    tmp_name = cli._prepare_config(
        config_path,
        seed=99,
        steps=12,
        steps_key="rl_steps",
        n_jobs=8,
        num_threads=4,
        validate=True,
    )

    try:
        assert tmp_name is not None
        assert os.environ["CONFIG_FILE"] == tmp_name
        with open(tmp_name, "r", encoding="utf-8") as fh:
            dumped = yaml.safe_load(fh)

        assert dumped["seed"] == 99
        assert dumped["training"]["seed"] == 99
        assert dumped["rl_steps"] == 12
        assert dumped["n_jobs"] == 8
        assert dumped["num_threads"] == 4
        assert dumped["validate"] is True

        assert config_path.read_text() == "seed: 1\n"
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
