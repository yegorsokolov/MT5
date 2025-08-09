"""Utilities to export trained models to ONNX with optional quantization."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional
    pd = None
try:  # pragma: no cover - torch optional
    import torch
except Exception:
    torch = None
import onnxruntime as ort
from lightgbm import LGBMClassifier
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import torch as torch_module

EXPORT_DIR = Path(__file__).resolve().parent / "exported"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _to_numpy(sample: Any) -> np.ndarray:
    if isinstance(sample, np.ndarray):
        return sample.astype(np.float32)
    if isinstance(sample, pd.DataFrame):
        return sample.to_numpy(dtype=np.float32)
    if isinstance(sample, torch.Tensor):
        return sample.detach().cpu().numpy().astype(np.float32)
    return np.asarray(sample, dtype=np.float32)


def export_lightgbm(
    model: LGBMClassifier | Any,
    sample: Any,
    export_dir: Path | None = None,
    validate: bool = True,
) -> Path:
    """Export a LightGBM model to ONNX with dynamic quantization.

    Parameters
    ----------
    model: trained ``LGBMClassifier`` or underlying booster.
    sample: sample input used to define the input shape.
    export_dir: directory to save ONNX artifacts.
    validate: whether to validate ONNX predictions against the original model.

    Returns
    -------
    Path to the quantized ONNX model.
    """

    export_dir = EXPORT_DIR if export_dir is None else export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    n_features = _to_numpy(sample).shape[1]
    booster = model.booster_ if hasattr(model, "booster_") else model
    onx = convert_lightgbm(
        booster,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        zipmap=False,
    )
    model_path = export_dir / "model_lightgbm.onnx"
    with model_path.open("wb") as f:
        f.write(onx.SerializeToString())
    quant_path = export_dir / "model_lightgbm_quant.onnx"
    try:  # pragma: no cover - quantization optional
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(model_path), str(quant_path), weight_type=QuantType.QInt8)
        onnx_to_check = quant_path
    except Exception:
        onnx_to_check = model_path

    if validate:
        sess = ort.InferenceSession(str(onnx_to_check))
        input_array = _to_numpy(sample)
        ort_outputs = sess.run(None, {"input": input_array})
        pred = (
            ort_outputs[1][:, 1]
            if len(ort_outputs) > 1
            else ort_outputs[0].ravel()
        )
        native = booster.predict(input_array)
        if native.ndim > 1:
            native = native[:, 1]
        if not np.allclose(pred, native, atol=1e-4):
            raise ValueError("ONNX LightGBM predictions differ from original model")
    return onnx_to_check


def export_pytorch(
    model: Any,
    sample: Any,
    export_dir: Path | None = None,
    validate: bool = True,
) -> Path:
    """Export a PyTorch model to ONNX with dynamic quantization."""
    if torch is None:  # pragma: no cover - requires torch
        raise ImportError("torch is required for export_pytorch")
    export_dir = EXPORT_DIR if export_dir is None else export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu").eval()
    sample_tensor = torch.tensor(_to_numpy(sample))
    model_path = export_dir / "model_torch.onnx"
    torch.onnx.export(
        model_cpu,
        sample_tensor,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    quant_path = export_dir / "model_torch_quant.onnx"
    try:  # pragma: no cover - quantization optional
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(model_path), str(quant_path), weight_type=QuantType.QInt8)
        onnx_to_check = quant_path
    except Exception:
        onnx_to_check = model_path

    if validate:
        sess = ort.InferenceSession(str(onnx_to_check))
        input_array = _to_numpy(sample)
        ort_pred = sess.run(None, {"input": input_array})[0]
        torch_pred = model_cpu(sample_tensor).detach().numpy()
        if not np.allclose(ort_pred, torch_pred, atol=1e-4):
            raise ValueError("ONNX PyTorch predictions differ from original model")
    return onnx_to_check

__all__ = ["export_lightgbm", "export_pytorch"]
