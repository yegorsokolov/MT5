import base64
import os
import sys
import types
import pickle
from pathlib import Path

import pytest

crypto_stub = types.SimpleNamespace(
    _load_key=lambda *a, **k: b"",
    encrypt=lambda data, key: data,
    decrypt=lambda blob, key: blob,
)
sys.modules.setdefault("crypto_utils", crypto_stub)
import joblib as _joblib

joblib_stub = types.SimpleNamespace(
    dump=_joblib.dump,
    load=_joblib.load,
    dumps=lambda obj: pickle.dumps(obj),
    loads=lambda data: pickle.loads(data),
)
sys.modules["joblib"] = joblib_stub
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mt5 import state_manager as sm


def _setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.delenv("CHECKPOINT_TIMESTAMP", raising=False)
    key = base64.b64encode(b"0" * 32).decode()
    monkeypatch.setenv("CHECKPOINT_AES_KEY", key)


def test_corruption_detected(tmp_path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    path = sm.save_checkpoint({"foo": "bar"}, 1)
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        data[0] ^= 0xFF
        f.seek(0)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    with pytest.raises(sm.StateCorruptionError):
        sm.load_latest_checkpoint(directory=str(tmp_path))
