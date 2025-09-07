import numpy as np
import torch
import importlib.util
from dataclasses import dataclass
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "quantize", Path(__file__).resolve().parents[1] / "models" / "quantize.py"
)
quantize_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(quantize_mod)  # type: ignore
apply_quantization = quantize_mod.apply_quantization

spec = importlib.util.spec_from_file_location(
    "model_registry", Path(__file__).resolve().parents[1] / "model_registry.py"
)
mr = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mr)  # type: ignore
ModelRegistry = mr.ModelRegistry
ModelVariant = mr.ModelVariant
ResourceCapabilities = mr.ResourceCapabilities

@dataclass
class DummyMonitor:
    capabilities: ResourceCapabilities

    def __post_init__(self) -> None:
        self.capability_tier = self.capabilities.capability_tier()

    def start(self) -> None:  # pragma: no cover
        pass


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def predict(self, x):
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32)
            return self.forward(t).numpy()


def test_registry_quantized_predictions_close(tmp_path):
    torch.manual_seed(0)
    model = TinyNet().eval()
    x = torch.randn(5, 4)
    ref = model.predict(x)
    qmodel = apply_quantization(model, bits=8)
    q_path = tmp_path / "tinynet_quantized.pkl"
    base_path = tmp_path / "tinynet.pkl"
    import joblib

    joblib.dump(model, base_path)
    joblib.dump(qmodel, q_path)

    monitor = DummyMonitor(ResourceCapabilities(1, 1, False, gpu_count=0))
    registry = ModelRegistry(monitor, auto_refresh=False)
    variant = ModelVariant(
        name="tinynet",
        requirements=ResourceCapabilities(1, 1, False, gpu_count=0),
        quantized="tinynet_quantized",
        weights=base_path,
        quantized_weights=q_path,
        quantized_requirements=ResourceCapabilities(1, 1, False, gpu_count=0),
    )
    baseline = ModelVariant(
        name="baseline",
        requirements=ResourceCapabilities(1, 1, False, gpu_count=0),
    )
    registry.register_variants("tiny", [variant, baseline])

    assert registry.get("tiny") == "tinynet_quantized"
    preds = registry.predict("tiny", x)
    assert np.allclose(preds, ref, atol=1e-1)
