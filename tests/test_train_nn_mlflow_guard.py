import base64
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

import pytest


def _prepare_mlflow(monkeypatch):
    if "requests" not in sys.modules:
        requests_stub = types.ModuleType("requests")
        requests_stub.__spec__ = importlib.machinery.ModuleSpec("requests", loader=None)
        monkeypatch.setitem(sys.modules, "requests", requests_stub)
    calls = {"start": 0, "end": 0}

    def _start_run(experiment: str, cfg):
        calls["start"] += 1
        assert experiment == "training_nn"
        assert cfg is not None
        return True

    def _end_run():
        calls["end"] += 1

    mlflow_module = types.ModuleType("analytics.mlflow_client")
    mlflow_module.start_run = _start_run
    mlflow_module.end_run = _end_run
    mlflow_module.log_param = lambda *args, **kwargs: None
    mlflow_module.log_metric = lambda *args, **kwargs: None
    mlflow_module.log_dict = lambda *args, **kwargs: None
    mlflow_module.log_artifact = lambda *args, **kwargs: None
    mlflow_module.log_artifacts = lambda *args, **kwargs: None

    analytics_pkg = types.ModuleType("analytics")
    analytics_pkg.mlflow_client = mlflow_module
    analytics_pkg.__path__ = []

    monkeypatch.setitem(sys.modules, "analytics", analytics_pkg)
    monkeypatch.setitem(sys.modules, "analytics.mlflow_client", mlflow_module)

    mlflow_stub = types.ModuleType("mlflow")
    mlflow_stub.active_run = lambda: None
    mlflow_stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)
    return calls


class _Dummy:
    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, _name):
        return self

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False


def _prepare_module(monkeypatch):
    original_spec = importlib.util.spec_from_file_location

    class _SecretLoader:
        def create_module(self, spec):  # type: ignore[override]
            return types.ModuleType(spec.name)

        def exec_module(self, module):  # type: ignore[override]
            class _SecretManager:
                def get_secret(self, name: str, default=None):
                    if default is not None:
                        return default
                    return base64.b64encode(b"0" * 32).decode()

            module.SecretManager = _SecretManager

    def _spec_from_file_location(name, location, *args, **kwargs):
        if name == "_secret_mgr":
            return importlib.machinery.ModuleSpec(name, _SecretLoader())
        return original_spec(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", _spec_from_file_location)

    class _PrefixStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def __init__(self, prefixes: tuple[str, ...]):
            self.prefixes = prefixes

        def find_spec(self, fullname, path=None, target=None):  # type: ignore[override]
            if any(
                fullname == prefix.rstrip(".") or fullname.startswith(prefix)
                for prefix in self.prefixes
            ):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):  # type: ignore[override]
            return types.ModuleType(spec.name)

        def exec_module(self, module):  # type: ignore[override]
            module.__dict__.setdefault("__path__", [])

            dummy = _Dummy()

            def _getattr(_name):
                return dummy

            module.__getattr__ = _getattr  # type: ignore[attr-defined]

    finder = _PrefixStubFinder((
        "analysis.",
        "models.",
        "data.",
        "services.",
        "training.",
        "utils.resource_monitor",
    ))
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    config_stub = types.ModuleType("mt5.config_models")
    config_stub.AppConfig = object
    config_stub.ConfigError = RuntimeError
    monkeypatch.setitem(sys.modules, "mt5.config_models", config_stub)

    cross_modal_stub = types.ModuleType("models.cross_modal_classifier")
    cross_modal_stub.CrossModalClassifier = object
    monkeypatch.setitem(sys.modules, "models.cross_modal_classifier", cross_modal_stub)

    prob_cal_stub = types.ModuleType("analysis.prob_calibration")
    prob_cal_stub.ProbabilityCalibrator = object
    prob_cal_stub.log_reliability = lambda *args, **kwargs: None
    prob_cal_stub.CalibratedModel = object
    monkeypatch.setitem(sys.modules, "analysis.prob_calibration", prob_cal_stub)

    orchestrator_stub = types.ModuleType("core.orchestrator")

    class _StubOrchestrator:
        @staticmethod
        def start():
            return object()

    orchestrator_stub.Orchestrator = _StubOrchestrator
    monkeypatch.setitem(sys.modules, "core.orchestrator", orchestrator_stub)

    module = importlib.import_module("mt5.train_nn")
    monkeypatch.setattr(module, "ensure_orchestrator_started", lambda: None)
    mlflow_module = sys.modules["analytics.mlflow_client"]
    monkeypatch.setattr(module.mlflow, "start_run", mlflow_module.start_run, raising=False)
    monkeypatch.setattr(module.mlflow, "end_run", mlflow_module.end_run, raising=False)
    return module


def test_train_nn_mlflow_end_run_on_success(monkeypatch):
    calls = _prepare_mlflow(monkeypatch)
    module = _prepare_module(monkeypatch)

    def _impl(*_args, **_kwargs):
        return 42.0

    monkeypatch.setattr(module, "_main_impl", _impl)

    result = module.main(0, 1, {"foo": "bar"})

    assert result == 42.0
    assert calls == {"start": 1, "end": 1}


def test_train_nn_mlflow_end_run_on_error(monkeypatch):
    calls = _prepare_mlflow(monkeypatch)
    module = _prepare_module(monkeypatch)

    class _Boom(Exception):
        pass

    def _impl(*_args, **_kwargs):
        raise _Boom("boom")

    monkeypatch.setattr(module, "_main_impl", _impl)

    with pytest.raises(_Boom):
        module.main(0, 1, {"foo": "bar"})

    assert calls == {"start": 1, "end": 1}
