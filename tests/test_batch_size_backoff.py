import ast
import pathlib
import types
import logging
import sys
import importlib.util
sys.modules.setdefault("analytics.metrics_store", types.SimpleNamespace(record_metric=lambda *a, **k: None))

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("resource_monitor", ROOT / "utils" / "resource_monitor.py")
rm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm_mod)  # type: ignore
monitor = rm_mod.monitor
source = (ROOT / "train_nn.py").read_text()
module = ast.parse(source)
func_src = None
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == "batch_size_backoff":
        func_src = ast.get_source_segment(source, node)
        break
assert func_src is not None

class _Cuda:
    OutOfMemoryError = RuntimeError
    def empty_cache(self):
        pass

class _Process:
    def memory_info(self):
        return types.SimpleNamespace(rss=0)

psutil_stub = types.SimpleNamespace(Process=_Process)
from typing import Callable, TypeVar
torch_stub = types.SimpleNamespace(cuda=_Cuda())
ns = {
    "psutil": psutil_stub,
    "torch": torch_stub,
    "monitor": monitor,
    "logging": logging,
    "logger": types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    "Callable": Callable,
    "TypeVar": TypeVar,
    "T": TypeVar("T"),
}
exec(func_src, ns)
batch_size_backoff = ns["batch_size_backoff"]


def test_batch_size_backoff(monkeypatch):
    # simulate low memory environment
    monkeypatch.setattr(monitor, "capabilities", types.SimpleNamespace(memory_gb=1))
    attempts = {"n": 0}

    def fake_train(bs, ebs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        assert bs == 32  # 64 -> 32 after backoff
        return bs

    result = batch_size_backoff({}, fake_train)
    assert result == 32
    assert attempts["n"] == 2
