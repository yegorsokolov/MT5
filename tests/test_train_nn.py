import importlib
import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch dependency unavailable",
)
def test_train_nn_module_exposes_main():
    module = importlib.import_module("train_nn")
    assert hasattr(module, "main")
