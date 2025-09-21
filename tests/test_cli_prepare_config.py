import copy
import os
import sys
import types
from pathlib import Path
from typing import Optional

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

    tmp_name, previous_config, env_modified = cli._prepare_config(
        config_path,
        seed=99,
        steps=12,
        steps_key="rl_steps",
        n_jobs=8,
        num_threads=4,
        validate=True,
    )

    try:
        assert previous_config == previous
        assert env_modified is True
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


def test_train_cmd_resets_config_env_between_invocations(tmp_path, monkeypatch):
    ensure_real_yaml()

    real_config = tmp_path / "real.yaml"
    real_config.write_text("seed: 1\n")
    other_config = tmp_path / "other.yaml"
    other_config.write_text("seed: 2\n")

    monkeypatch.setenv("CONFIG_FILE", str(real_config))

    seen_paths: list[Optional[str]] = []

    def fake_load_config() -> dict:
        seen_paths.append(os.environ.get("CONFIG_FILE"))
        return {"seed": 1, "training": {"seed": 1}}

    monkeypatch.setattr(cli, "load_config", fake_load_config)

    train_paths: list[Optional[str]] = []

    def dummy_train_main() -> None:
        train_paths.append(os.environ.get("CONFIG_FILE"))

    monkeypatch.setitem(sys.modules, "train", types.SimpleNamespace(main=dummy_train_main))

    args_first = types.SimpleNamespace(
        config=other_config,
        seed=None,
        n_jobs=None,
        validate=None,
    )
    cli.train_cmd(args_first)

    assert seen_paths == [str(other_config)]
    assert train_paths == [str(other_config)]
    assert os.environ.get("CONFIG_FILE") == str(real_config)

    args_second = types.SimpleNamespace(
        config=None,
        seed=303,
        n_jobs=None,
        validate=None,
    )
    cli.train_cmd(args_second)

    assert len(seen_paths) == 2
    assert seen_paths[1] == str(real_config)
    assert len(train_paths) == 2
    assert train_paths[1] not in {None, str(real_config), str(other_config)}
    assert os.environ.get("CONFIG_FILE") == str(real_config)
