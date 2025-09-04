import os
import sys
import types
import contextlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import analytics.mlflow_client as mlflow_client


def test_start_run_and_log_artifact(monkeypatch, tmp_path):
    calls = {}

    def set_tracking_uri(uri):
        calls["uri"] = uri

    def set_experiment(exp):
        calls["exp"] = exp

    def start_run():
        calls["started"] = True

    def log_dict(cfg, name):
        calls["dict"] = (cfg, name)

    def log_param(k, v):
        calls.setdefault("params", {})[k] = v

    def log_metric(k, v, step=None):
        calls.setdefault("metrics", {})[k] = (v, step)

    def log_artifact(path):
        calls["artifact"] = path

    def end_run():
        calls["ended"] = True

    mlflow_stub = types.SimpleNamespace(
        set_tracking_uri=set_tracking_uri,
        set_experiment=set_experiment,
        start_run=start_run,
        log_dict=log_dict,
        log_param=log_param,
        log_metric=log_metric,
        log_artifact=log_artifact,
        end_run=end_run,
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)
    # reload module to pick up stub
    import importlib

    importlib.reload(mlflow_client)

    cfg = {
        "mlflow": {
            "tracking_uri": "http://remote:5000",
            "username": "user",
            "password": "pass",
        }
    }
    mlflow_client.start_run("test", cfg)
    mlflow_client.log_param("alpha", 1)
    mlflow_client.log_metric("loss", 0.5, step=1)
    p = tmp_path / "a.txt"
    p.write_text("x")
    mlflow_client.log_artifact(str(p))
    mlflow_client.end_run()

    assert calls["uri"] == "http://remote:5000"
    assert calls["exp"] == "test"
    assert calls["dict"][0] == cfg
    assert calls["params"]["alpha"] == 1
    assert calls["metrics"]["loss"] == (0.5, 1)
    assert calls["artifact"] == str(p)
    assert calls["ended"] is True
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "user"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "pass"
