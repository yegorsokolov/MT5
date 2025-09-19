from __future__ import annotations

import argparse
import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_cli(monkeypatch: pytest.MonkeyPatch):
    """Import the CLI module with lightweight stubs."""

    utils_stub = types.ModuleType("utils")
    utils_stub.ensure_environment = lambda: None
    utils_stub.load_config = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "utils", utils_stub)
    resource_monitor_stub = types.ModuleType("utils.resource_monitor")
    monitor_stub = types.SimpleNamespace(
        capabilities=types.SimpleNamespace(capability_tier=lambda: "lite", ddp=lambda: False),
        capability_tier="lite",
        start=lambda: None,
        stop=lambda: None,
        subscribe=lambda: types.SimpleNamespace(get=lambda: None),
        create_task=lambda *a, **k: None,
    )
    resource_monitor_stub.monitor = monitor_stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.resource_monitor", resource_monitor_stub)
    monkeypatch.delitem(sys.modules, "cli", raising=False)
    return importlib.import_module("cli")


def _load_train_rl(monkeypatch: pytest.MonkeyPatch):
    """Load ``train_rl`` with heavy dependencies stubbed out."""

    monkeypatch.delitem(sys.modules, "train_rl", raising=False)

    mlflow_stub = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_artifact=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        __spec__=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)

    log_utils_stub = types.SimpleNamespace(
        setup_logging=lambda *a, **k: None,
        log_exceptions=lambda f: f,
        __spec__=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "log_utils", log_utils_stub)

    class _NP(types.SimpleNamespace):
        float32 = float
        ndarray = type("FakeNDArray", (), {})

        def __getattr__(self, name):  # pragma: no cover - simple fallback
            def _(*args, **kwargs):
                if name == "linspace" and len(args) >= 3:
                    start, stop, num = args[:3]
                    return [start + (stop - start) * i / (num - 1) for i in range(num)]
                return 0

            return _

    np_stub = _NP(
        random=types.SimpleNamespace(seed=lambda *a, **k: None, randn=lambda *a, **k: 0, normal=lambda *a, **k: 0)
    )
    np_stub.isscalar = lambda obj: isinstance(obj, (int, float))
    np_stub.bool_ = bool
    monkeypatch.setitem(sys.modules, "numpy", np_stub)

    class _PD(types.SimpleNamespace):
        def __getattr__(self, name):  # pragma: no cover - simple fallback
            return lambda *a, **k: None

    pd_stub = _PD(DataFrame=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "pandas", pd_stub)

    class _FakeTensor:
        def __init__(self, data=None, **_kwargs):
            self.data = data

        def item(self):
            if isinstance(self.data, (int, float)):
                return float(self.data)
            return 0.0

        def backward(self):  # pragma: no cover - used in fallback paths
            return None

        def unsqueeze(self, *_args, **_kwargs):
            return self

        def dim(self):  # pragma: no cover - simple placeholder
            return 1

        def cpu(self):  # pragma: no cover - simple placeholder
            return self

        def numpy(self):  # pragma: no cover - simple placeholder
            return self.data

    def _fake_tensor_factory(*args, **kwargs):
        if args:
            return _FakeTensor(args[0], **kwargs)
        return _FakeTensor(**kwargs)

    torch_stub = types.ModuleType("torch")
    torch_stub.manual_seed = lambda *a, **k: None
    torch_stub.Tensor = _FakeTensor
    torch_stub.tensor = _fake_tensor_factory
    torch_stub.as_tensor = _fake_tensor_factory
    torch_stub.softmax = lambda *a, **k: _FakeTensor(0.0)
    torch_stub.cat = lambda tensors, **_k: _FakeTensor([getattr(t, "data", None) for t in tensors])
    torch_stub.no_grad = types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *exc: None)
    torch_stub.float32 = "float32"
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *a, **k: None,
        device_count=lambda: 0,
        set_device=lambda *a, **k: None,
    )
    torch_stub.__spec__ = types.SimpleNamespace()

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):  # pragma: no cover - simple placeholder
            return []

        def __call__(self, *args, **kwargs):
            forward = getattr(self, "forward", None)
            if callable(forward):
                return forward(*args, **kwargs)
            return None

    class _Linear(_Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x):  # pragma: no cover - simple placeholder
            return x

    class _ReLU(_Module):
        def forward(self, x):  # pragma: no cover - simple placeholder
            return x

    class _Sequential(_Module):
        def __init__(self, *modules):
            super().__init__()
            self._modules = modules

        def forward(self, x):  # pragma: no cover - simple placeholder
            for mod in self._modules:
                x = mod(x)
            return x

    class _Loss:
        def __call__(self, *args, **kwargs):  # pragma: no cover - simple placeholder
            return _FakeTensor(0.0)

    torch_nn_stub = types.ModuleType("torch.nn")
    torch_nn_stub.Module = _Module
    torch_nn_stub.Linear = _Linear
    torch_nn_stub.ReLU = _ReLU
    torch_nn_stub.Sequential = _Sequential
    torch_nn_stub.MSELoss = _Loss
    torch_nn_stub.CrossEntropyLoss = _Loss
    torch_nn_stub.functional = types.SimpleNamespace(mse_loss=lambda *a, **k: _FakeTensor(0.0))

    torch_nn_parallel_stub = types.ModuleType("torch.nn.parallel")

    class _DDP:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            pass

    torch_nn_parallel_stub.DistributedDataParallel = _DDP

    class _Optimizer:
        def __init__(self, *args, **kwargs):
            self.param_groups = [{}]

        def zero_grad(self):  # pragma: no cover - simple placeholder
            return None

        def step(self):  # pragma: no cover - simple placeholder
            return None

        def get_lr(self):  # pragma: no cover - simple placeholder
            return 0.0

    torch_optim_stub = types.ModuleType("torch.optim")
    torch_optim_stub.Adam = lambda *a, **k: _Optimizer()

    torch_stub.nn = torch_nn_stub
    torch_stub.optim = torch_optim_stub

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn_stub)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", torch_nn_stub.functional)
    monkeypatch.setitem(sys.modules, "torch.nn.parallel", torch_nn_parallel_stub)
    monkeypatch.setitem(sys.modules, "torch.optim", torch_optim_stub)
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_utils_data_stub = types.ModuleType("torch.utils.data")
    torch_utils_data_stub.DataLoader = object
    torch_utils_data_stub.TensorDataset = object
    torch_utils_stub.data = torch_utils_data_stub
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_stub)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data_stub)

    gym_stub = types.SimpleNamespace(
        Env=object,
        Wrapper=object,
        spaces=types.SimpleNamespace(Box=object),
    )
    monkeypatch.setitem(sys.modules, "gym", gym_stub)

    utils_stub = types.SimpleNamespace(load_config=lambda: {})
    monkeypatch.setitem(sys.modules, "utils", utils_stub)

    state_stub = types.SimpleNamespace(
        save_checkpoint=lambda *a, **k: None,
        load_latest_checkpoint=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "state_manager", state_stub)

    history_stub = types.SimpleNamespace(
        load_history_parquet=lambda *a, **k: None,
        save_history_parquet=lambda *a, **k: None,
        load_history_config=lambda *a, **k: pd_stub.DataFrame(),
    )
    def _make_features(df, **_kwargs):
        return df

    features_stub = types.SimpleNamespace(make_features=_make_features)
    features_module = types.ModuleType("features")
    features_module.make_features = _make_features
    data_pkg = types.ModuleType("data")
    data_pkg.__path__ = []  # mark as package
    monkeypatch.setitem(sys.modules, "data", data_pkg)
    monkeypatch.setitem(sys.modules, "data.history", history_stub)
    monkeypatch.setitem(sys.modules, "data.features", features_stub)
    monkeypatch.setitem(sys.modules, "features", features_module)
    events_stub = types.SimpleNamespace(get_events=lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "data.events", events_stub)

    models_stub = types.SimpleNamespace(
        model_store=types.SimpleNamespace(save_model=lambda *a, **k: "0"),
        graph_net=types.SimpleNamespace(GraphNet=object),
    )
    monkeypatch.setitem(sys.modules, "models", models_stub)
    monkeypatch.setitem(sys.modules, "models.model_store", models_stub.model_store)
    monkeypatch.setitem(sys.modules, "models.graph_net", models_stub.graph_net)

    class _MarketSim:
        def __init__(self, *args, **kwargs):
            self.seq_len = kwargs.get("seq_len", 0)

        def perturb(self, *args, **kwargs):  # pragma: no cover - simple placeholder
            return None

    grad_monitor_stub = types.SimpleNamespace(
        GradientMonitor=lambda *a, **k: types.SimpleNamespace(
            track=lambda *args, **kwargs: ("stable", {}),
            plot=lambda *args, **kwargs: None,
        ),
        GradMonitorCallback=object,
    )

    analysis_stub = types.SimpleNamespace(
        regime_detection=types.SimpleNamespace(periodic_reclassification=lambda df, **k: df),
        model_card=types.SimpleNamespace(log_model_card=lambda *a, **k: None),
        market_simulator=types.SimpleNamespace(
            AdversarialMarketSimulator=_MarketSim,
            generate_stress_scenarios=lambda *a, **k: {},
        ),
        grad_monitor=grad_monitor_stub,
        regime_hmm=types.SimpleNamespace(fit_regime_hmm=lambda *a, **k: None),
        inference_latency=types.SimpleNamespace(InferenceLatency=object),
    )
    monkeypatch.setitem(sys.modules, "analysis", analysis_stub)
    monkeypatch.setitem(sys.modules, "analysis.regime_detection", analysis_stub.regime_detection)
    monkeypatch.setitem(sys.modules, "analysis.model_card", analysis_stub.model_card)
    monkeypatch.setitem(
        sys.modules,
        "analysis.market_simulator",
        analysis_stub.market_simulator,
    )
    monkeypatch.setitem(
        sys.modules,
        "analysis.grad_monitor",
        grad_monitor_stub,
    )
    monkeypatch.setitem(
        sys.modules,
        "analysis.regime_hmm",
        analysis_stub.regime_hmm,
    )
    monkeypatch.setitem(
        sys.modules,
        "analysis.inference_latency",
        analysis_stub.inference_latency,
    )

    metrics_stub = types.SimpleNamespace(
        record_metric=lambda *a, **k: None,
        TS_PATH=Path("metrics.parquet"),
        __spec__=types.SimpleNamespace(),
    )
    analytics_stub = types.SimpleNamespace(metrics_store=metrics_stub, __path__=[], __spec__=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "analytics", analytics_stub)
    monkeypatch.setitem(sys.modules, "analytics.metrics_store", metrics_stub)
    model_registry_stub = types.ModuleType("model_registry")
    model_registry_stub.register_policy = lambda *a, **k: None
    model_registry_stub.save_model = lambda *a, **k: None
    model_registry_stub.get_policy_path = lambda *a, **k: Path("policy.zip")
    monkeypatch.setitem(sys.modules, "model_registry", model_registry_stub)
    strategy_pkg = types.ModuleType("strategy")
    self_review_module = types.ModuleType("strategy.self_review")
    self_review_module.self_review_strategy = lambda prompt, *_a, **_k: prompt
    strategy_pkg.self_review = self_review_module
    monkeypatch.setitem(sys.modules, "strategy", strategy_pkg)
    monkeypatch.setitem(sys.modules, "strategy.self_review", self_review_module)

    sb3_common = types.SimpleNamespace(
        vec_env=types.SimpleNamespace(SubprocVecEnv=object, DummyVecEnv=object),
        evaluation=types.SimpleNamespace(evaluate_policy=lambda *a, **k: None),
    )
    sb3_stub = types.SimpleNamespace(PPO=object, SAC=object, A2C=object, common=sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_stub)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", sb3_common.vec_env)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.evaluation", sb3_common.evaluation)

    sb3_contrib_stub = types.SimpleNamespace(
        TRPO=object,
        RecurrentPPO=object,
        qrdqn=types.SimpleNamespace(QRDQN=object),
    )
    monkeypatch.setitem(sys.modules, "sb3_contrib", sb3_contrib_stub)
    monkeypatch.setitem(sys.modules, "sb3_contrib.qrdqn", sb3_contrib_stub.qrdqn)

    plugins_stub = types.SimpleNamespace()
    rl_risk_stub = types.SimpleNamespace(RiskEnv=object)
    monkeypatch.setitem(sys.modules, "plugins", plugins_stub)
    monkeypatch.setitem(sys.modules, "plugins.rl_risk", rl_risk_stub)

    ray_utils_stub = types.SimpleNamespace(
        init=lambda *a, **k: None,
        shutdown=lambda *a, **k: None,
        cluster_available=lambda: False,
        submit=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "ray_utils", ray_utils_stub)

    event_store_stub = types.SimpleNamespace(EventStore=types.SimpleNamespace)
    monkeypatch.setitem(sys.modules, "event_store", event_store_stub)

    spec = importlib.util.spec_from_file_location(
        "train_rl", Path(__file__).resolve().parents[1] / "train_rl.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "train_rl", module)
    monkeypatch.setattr(
        module,
        "init_logging",
        lambda: logging.getLogger("test_train_rl"),
        raising=False,
    )
    return module


def test_train_rl_cli_missing_torch(monkeypatch: pytest.MonkeyPatch):
    """CLI should surface a helpful error when PyTorch is unavailable."""

    cli_mod = _load_cli(monkeypatch)
    train_rl_mod = _load_train_rl(monkeypatch)

    args = argparse.Namespace(config=None, seed=None, steps=None, validate=False)
    monkeypatch.setattr(cli_mod, "_prepare_config", lambda *a, **k: None)
    monkeypatch.setattr(train_rl_mod, "_TORCH_AVAILABLE", False, raising=False)
    monkeypatch.setattr(
        train_rl_mod,
        "_TORCH_IMPORT_ERROR",
        ImportError("No module named 'torch'"),
        raising=False,
    )
    monkeypatch.setattr(train_rl_mod, "torch", None, raising=False)

    with pytest.raises(RuntimeError, match="PyTorch"):
        cli_mod.train_rl_cmd(args)


def test_train_rl_launch_cpu_only(monkeypatch: pytest.MonkeyPatch):
    """CPU-only environments should fall back without touching CUDA APIs."""

    train_rl_mod = _load_train_rl(monkeypatch)

    called: list[tuple[int, int, dict]] = []

    def _dummy_main(rank: int, world_size: int, cfg: dict) -> float:
        called.append((rank, world_size, cfg))
        return 3.14

    torch_stub = types.SimpleNamespace(
        manual_seed=lambda *a, **k: None,
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=lambda *a, **k: None,
            device_count=lambda: 0,
            set_device=lambda *a, **k: None,
        ),
    )

    monkeypatch.setattr(train_rl_mod, "torch", torch_stub, raising=False)
    monkeypatch.setattr(train_rl_mod, "_TORCH_AVAILABLE", True, raising=False)
    monkeypatch.setattr(train_rl_mod, "_CUDA_MODULE_AVAILABLE", True, raising=False)
    monkeypatch.setattr(train_rl_mod, "_TORCH_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(train_rl_mod, "main", _dummy_main)
    monkeypatch.setattr(train_rl_mod, "cluster_available", lambda: False)
    monkeypatch.setattr(
        train_rl_mod,
        "mp",
        types.SimpleNamespace(spawn=lambda *a, **k: None),
        raising=False,
    )

    result = train_rl_mod.launch({"ddp": False})

    assert result == pytest.approx(3.14)
    assert called == [(0, 1, {"ddp": False})]


@pytest.mark.parametrize(
    "cuda_module_available, message",
    [
        (False, "CUDA support is not available"),
        (True, "CUDA kernels are unavailable"),
    ],
)
def test_train_rl_launch_ddp_requires_cuda(
    monkeypatch: pytest.MonkeyPatch, cuda_module_available: bool, message: str
):
    """Distributed training should fail fast when CUDA kernels are unavailable."""

    train_rl_mod = _load_train_rl(monkeypatch)

    torch_stub = types.SimpleNamespace(manual_seed=lambda *a, **k: None)
    if cuda_module_available:
        torch_stub.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=lambda *a, **k: None,
            device_count=lambda: 0,
            set_device=lambda *a, **k: None,
        )

    monkeypatch.setattr(train_rl_mod, "torch", torch_stub, raising=False)
    monkeypatch.setattr(train_rl_mod, "_TORCH_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        train_rl_mod,
        "_CUDA_MODULE_AVAILABLE",
        cuda_module_available,
        raising=False,
    )
    monkeypatch.setattr(train_rl_mod, "_TORCH_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(train_rl_mod, "cluster_available", lambda: False)

    with pytest.raises(RuntimeError, match=message):
        train_rl_mod.launch({"ddp": True})
