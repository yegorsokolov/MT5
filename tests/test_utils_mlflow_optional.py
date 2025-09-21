"""Regression tests ensuring utils works when MLflow is optional."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _preserve_utils_modules() -> dict[str, object]:
    """Return the currently loaded ``utils`` modules keyed by name."""

    return {
        name: module
        for name, module in sys.modules.items()
        if name == "utils" or name.startswith("utils.")
    }


def _clear_utils_modules() -> None:
    """Remove ``utils`` and its submodules from ``sys.modules``."""

    for name in [
        module_name
        for module_name in sys.modules
        if module_name == "utils" or module_name.startswith("utils.")
    ]:
        sys.modules.pop(name, None)


def test_utils_mlflow_optional(monkeypatch):
    """Sanity-check ``utils`` functions when MLflow cannot be imported."""

    saved_modules = _preserve_utils_modules()

    with monkeypatch.context() as mpatch:
        mpatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        sys.modules.pop("mlflow", None)
        mpatch.setitem(sys.modules, "mlflow", None)
        sys.modules.pop("yaml", None)
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda *args, **kwargs: {}
        yaml_stub.safe_dump = lambda *args, **kwargs: None
        mpatch.setitem(sys.modules, "yaml", yaml_stub)

        log_utils_stub = types.ModuleType("log_utils")
        log_utils_stub.LOG_DIR = Path("./logs")
        mpatch.setitem(sys.modules, "log_utils", log_utils_stub)

        class _BaseModel:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def model_dump(self):
                return dict(self.__dict__)

        pydantic_stub = types.ModuleType("pydantic")
        pydantic_stub.BaseModel = _BaseModel
        pydantic_stub.ValidationError = Exception
        mpatch.setitem(sys.modules, "pydantic", pydantic_stub)

        class _FileLock:
            def __init__(self, path):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        filelock_stub = types.ModuleType("filelock")
        filelock_stub.FileLock = _FileLock
        mpatch.setitem(sys.modules, "filelock", filelock_stub)

        class _ConfigError(Exception):
            pass

        class _AppConfig(_BaseModel):
            pass

        config_models_stub = types.ModuleType("config_models")
        config_models_stub.AppConfig = _AppConfig
        config_models_stub.ConfigError = _ConfigError
        mpatch.setitem(sys.modules, "config_models", config_models_stub)
        _clear_utils_modules()
        importlib.invalidate_caches()

        utils = importlib.import_module("utils")

        sanitized = utils.sanitize_config({"param": "value"})
        assert sanitized == {"param": "value"}

        with utils.mlflow_run("optional-mlflow", {"param": "value"}):
            pass

    _clear_utils_modules()
    sys.modules.update(saved_modules)
