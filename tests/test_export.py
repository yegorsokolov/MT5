import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import importlib
try:
    torch = importlib.import_module("torch")
except Exception:  # pragma: no cover - torch optional
    torch = None
from lightgbm import LGBMClassifier
import onnxruntime as ort
import importlib.util

module_path = Path(__file__).resolve().parents[1] / "models" / "export.py"
spec = importlib.util.spec_from_file_location("export", module_path)
export = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(export)
export_lightgbm = export.export_lightgbm
export_pytorch = export.export_pytorch


def test_export_lightgbm(tmp_path):
    X = np.random.rand(20, 4).astype(np.float32)
    y = (X.sum(axis=1) > 2).astype(int)
    model = LGBMClassifier(n_estimators=5, min_child_samples=1)
    model.fit(X, y)
    onnx_path = export_lightgbm(model, X[:1], tmp_path)
    sess = ort.InferenceSession(str(onnx_path))
    ort_pred = sess.run(None, {"input": X[:5]})[1][:, 1]
    native = model.predict_proba(X[:5])[:, 1]
    assert np.allclose(ort_pred, native, atol=1e-4)


def test_export_pytorch(tmp_path):
    if torch is None:
        pytest.skip("torch not available")
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 1)

        def forward(self, x):
            return torch.sigmoid(self.fc(x))

    model = Net()
    sample = torch.randn(1, 3)
    onnx_path = export_pytorch(model, sample, tmp_path)
    sess = ort.InferenceSession(str(onnx_path))
    inputs = torch.randn(5, 3)
    ort_pred = sess.run(None, {"input": inputs.numpy().astype(np.float32)})[0]
    torch_pred = model(inputs).detach().numpy()
    assert np.allclose(ort_pred, torch_pred, atol=1e-4)
