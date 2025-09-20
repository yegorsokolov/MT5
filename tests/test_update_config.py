import json
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

yaml_stub = types.ModuleType("yaml")


def _safe_load(stream, *_, **__):
    if hasattr(stream, "read"):
        content = stream.read()
    else:
        content = stream
    if not content:
        return {}
    return json.loads(content)


def _safe_dump(data, stream=None, *_, **__):
    text = json.dumps(data)
    if stream is None:
        return text
    stream.write(text)
    return None


yaml_stub.safe_load = _safe_load
yaml_stub.safe_dump = _safe_dump
yaml_stub.__spec__ = types.SimpleNamespace()
sys.modules["yaml"] = yaml_stub
yaml = yaml_stub

pydantic_stub = types.ModuleType("pydantic")


class ValidationError(Exception):
    pass


class BaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self):
        return {key: value for key, value in self.__dict__.items()}


pydantic_stub.ValidationError = ValidationError
pydantic_stub.BaseModel = BaseModel
pydantic_stub.__spec__ = types.SimpleNamespace()
sys.modules["pydantic"] = pydantic_stub

config_models_stub = types.ModuleType("config_models")


class ConfigError(ValueError):
    pass


class AppConfig(BaseModel):
    def __init__(self, **data):
        strategy = data.get("strategy") or {}
        symbols = strategy.get("symbols") if isinstance(strategy, dict) else None
        if not symbols:
            raise ValidationError("strategy symbols must be provided")
        super().__init__(**data)


config_models_stub.ConfigError = ConfigError
config_models_stub.AppConfig = AppConfig
config_models_stub.__spec__ = types.SimpleNamespace()
sys.modules["config_models"] = config_models_stub

filelock_stub = types.ModuleType("filelock")


class FileLock:
    def __init__(self, *_args, **_kwargs):
        self._locked = False

    def __enter__(self):
        self._locked = True
        return self

    def __exit__(self, *exc):
        self._locked = False
        return False


filelock_stub.FileLock = FileLock
filelock_stub.__spec__ = types.SimpleNamespace()
sys.modules["filelock"] = filelock_stub

sys.modules.pop("utils", None)
import utils


def test_update_config_preserves_secret_placeholder(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_data = {
        "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1},
        "mlflow": {"password": "secret://MLFLOW_PASS"},
        "plugin_cache_ttl": 60,
    }
    cfg_file.write_text(yaml.safe_dump(cfg_data))

    monkeypatch.setenv("CONFIG_FILE", str(cfg_file))
    monkeypatch.setenv("MLFLOW_PASS", "resolved-secret")

    utils.update_config("plugin_cache_ttl", 120, "Adjust TTL")

    saved_text = cfg_file.read_text()
    assert "resolved-secret" not in saved_text

    saved = yaml.safe_load(saved_text)
    assert saved["mlflow"]["password"] == "secret://MLFLOW_PASS"
    assert saved["plugin_cache_ttl"] == 120
