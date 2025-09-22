import ast
from pathlib import Path

import types

ROOT = Path(__file__).resolve().parents[1] / "mt5"


def _load_launch(path: Path):
    src = path.read_text()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "launch":
            module = ast.Module(body=[node], type_ignores=[])
            code = compile(module, path, "exec")
            ns: dict[str, types.Any] = {}
            exec(code, ns)
            return ns["launch"]
    raise ValueError("launch not found")


def _run_launch(file: str) -> int:
    launch = _load_launch(ROOT / file)
    calls = []

    def fake_submit(fn, *args, **kwargs):
        calls.append(1)
        return 0

    launch_globals = {
        "load_config": lambda: {},
        "cluster_available": lambda: True,
        "submit": fake_submit,
        "main": lambda *a, **k: 0,
        "AppConfig": object,
    }
    launch.__globals__.update(launch_globals)
    launch({"seed": 1})
    return len(calls)


def test_train_launch_distributed():
    assert _run_launch("train.py") > 0


def test_train_nn_launch_distributed():
    assert _run_launch("train_nn.py") > 0


def test_train_rl_launch_distributed():
    assert _run_launch("train_rl.py") > 0
