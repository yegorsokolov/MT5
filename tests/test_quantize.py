import torch
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "quantize", Path(__file__).resolve().parents[1] / "models" / "quantize.py"
)
quantize = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(quantize)  # type: ignore
apply_quantization = quantize.apply_quantization


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_apply_quantization_preserves_outputs() -> None:
    torch.manual_seed(0)
    model = TinyNet()
    x = torch.randn(5, 4)
    with torch.no_grad():
        ref = model(x)
    qmodel = apply_quantization(model, bits=8)
    qout = qmodel(x)
    assert torch.allclose(ref, qout, atol=1e-1)


def test_apply_quantization_with_pruning() -> None:
    torch.manual_seed(0)
    model = TinyNet()
    original_nonzero = int(model.fc1.weight.ne(0).sum())
    qmodel = apply_quantization(model, bits=8, prune_ratio=0.5)
    pruned_nonzero = int(qmodel.fc1.weight.ne(0).sum())
    assert pruned_nonzero <= original_nonzero
