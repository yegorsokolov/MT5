import pandas as pd
import numpy as np
from pathlib import Path

import dataset
import utils


def test_make_features_uses_cache(monkeypatch, tmp_path):
    data_dir = Path(dataset.__file__).resolve().parent / "data"
    if data_dir.exists():
        for f in data_dir.iterdir():
            f.unlink()
    else:
        data_dir.mkdir()

    monkeypatch.setattr(
        utils,
        "load_config",
        lambda: {
            "use_feature_cache": True,
            "use_atr": False,
            "use_donchian": False,
            "use_dask": False,
        },
    )

    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=60, freq="min"),
        "Bid": np.linspace(1.0, 2.0, 60),
        "Ask": np.linspace(1.0001, 2.0001, 60),
    })

    first = dataset.make_features(df)
    cache_file = data_dir / "features.duckdb"
    assert cache_file.exists()

    logs = []
    monkeypatch.setattr(dataset.logger, "info", lambda msg, *a: logs.append(msg % a if a else msg))
    second = dataset.make_features(df)
    assert any("Loading features from cache" in m for m in logs)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
        check_dtype=False,
    )

