import importlib.util
import sys
from pathlib import Path

import warnings
import pandas as pd
import pytest
import time
import numpy as np
from pandas.errors import PerformanceWarning

FEATURES_PATH = Path(__file__).resolve().parents[1] / "features" / "cross_asset.py"
spec = importlib.util.spec_from_file_location("cross_asset", FEATURES_PATH)
cross_asset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cross_asset)
add_cross_asset_features = cross_asset.add_cross_asset_features
DEFAULT_MAX_PAIRS = cross_asset.DEFAULT_MAX_PAIRS


def _baseline_add_cross_asset_features(
    df: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    required = {"Symbol", "Timestamp", "return"}
    if not required.issubset(df.columns):
        return df

    df = df.copy().sort_values("Timestamp")
    pivot = df.pivot(index="Timestamp", columns="Symbol", values="return")
    symbols = list(pivot.columns)

    cs_mean = pivot.mean(axis=1)
    cs_std = pivot.std(axis=1, ddof=0).replace(0, np.nan)
    rel_strength = pivot.sub(cs_mean, axis=0).div(cs_std, axis=0)

    for sym in symbols:
        ts_map = df.loc[df["Symbol"] == sym, "Timestamp"]
        df.loc[df["Symbol"] == sym, f"rel_strength_{sym}"] = ts_map.map(
            rel_strength[sym]
        ).fillna(0.0)

    for sym1 in symbols:
        for sym2 in symbols:
            if sym1 == sym2:
                continue

            corr_series = pivot[sym1].rolling(window).corr(pivot[sym2])
            ratio_series = (pivot[sym1] / pivot[sym2]).replace(
                [np.inf, -np.inf], np.nan
            )

            ts_map = df.loc[df["Symbol"] == sym1, "Timestamp"]
            df.loc[df["Symbol"] == sym1, f"corr_{sym1}_{sym2}"] = ts_map.map(
                corr_series
            ).fillna(0.0)
            df.loc[df["Symbol"] == sym1, f"relret_{sym1}_{sym2}"] = ts_map.map(
                ratio_series
            ).fillna(0.0)

    return df


def _sample_df():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=5, freq="D").tolist() * 2,
            "Symbol": ["AAA"] * 5 + ["BBB"] * 5,
            "mid": [1, 2, 3, 4, 5, 2, 1, 2, 3, 4],
        }
    )
    df["return"] = df.groupby("Symbol")["mid"].pct_change()
    return df


def test_cross_asset_features_creation():
    df = _sample_df()
    out = add_cross_asset_features(df, window=3)

    for col in [
        "corr_AAA_BBB",
        "corr_BBB_AAA",
        "relret_AAA_BBB",
        "relret_BBB_AAA",
    ]:
        assert col in out.columns

    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == df.shape[1] + 6

    wide = df.pivot(index="Timestamp", columns="Symbol", values="return")
    expected_corr = wide["AAA"].rolling(3).corr(wide["BBB"]).iloc[-1]
    expected_ratio = (wide["AAA"] / wide["BBB"]).iloc[-1]

    last_row = out[
        (out["Symbol"] == "AAA") & (out["Timestamp"] == df["Timestamp"].max())
    ]
    assert last_row["corr_AAA_BBB"].iloc[0] == pytest.approx(expected_corr)
    assert last_row["relret_AAA_BBB"].iloc[0] == pytest.approx(expected_ratio)


def test_cross_asset_whitelist():
    df = _sample_df()
    extra = df[df["Symbol"] == "AAA"].copy()
    extra["Symbol"] = "CCC"
    df = pd.concat([df, extra], ignore_index=True)

    out = add_cross_asset_features(df, window=3, whitelist=["AAA", "BBB"])

    assert "corr_AAA_BBB" in out.columns
    assert "corr_AAA_CCC" not in out.columns
    assert "corr_BBB_CCC" not in out.columns


def test_cross_asset_default_limit_bounds_columns_and_runtime():
    symbols = [f"S{i:03d}" for i in range(32)]
    periods = 40
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Timestamp": idx.repeat(len(symbols)),
            "Symbol": symbols * periods,
            "return": rng.standard_normal(periods * len(symbols)) / 100,
        }
    )

    start = time.perf_counter()
    limited = add_cross_asset_features(df.copy(), window=5)
    limited_time = time.perf_counter() - start

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        start = time.perf_counter()
        base = _baseline_add_cross_asset_features(df.copy(), window=5).fillna(0.0)
        baseline_time = time.perf_counter() - start

    pca_cols = [c for c in limited.columns if c.startswith("pair_pca_")]
    pair_cols_baseline = [
        c for c in base.columns if c.startswith("corr_") or c.startswith("relret_")
    ]

    assert len(pca_cols) == DEFAULT_MAX_PAIRS
    assert not any(
        c.startswith("corr_") or c.startswith("relret_") for c in limited.columns
    )
    assert len(pair_cols_baseline) > len(pca_cols)
    assert limited_time < baseline_time


def test_cross_asset_unlimited_matches_baseline():
    symbols = [f"S{i:03d}" for i in range(20)]
    periods = 35
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "Timestamp": idx.repeat(len(symbols)),
            "Symbol": symbols * periods,
            "return": rng.standard_normal(periods * len(symbols)) / 100,
        }
    )

    full = add_cross_asset_features(df.copy(), window=5, max_pairs=None).fillna(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        base = _baseline_add_cross_asset_features(df.copy(), window=5).fillna(0.0)

    pd.testing.assert_frame_equal(full.sort_index(axis=1), base.sort_index(axis=1))


def test_cross_asset_top_k_limits_pairs_and_time():
    symbols = [f"S{i:03d}" for i in range(50)]
    periods = 30
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "Timestamp": idx.repeat(len(symbols)),
            "Symbol": symbols * periods,
            "return": rng.standard_normal(periods * len(symbols)) / 100,
        }
    )

    start = time.perf_counter()
    limited = add_cross_asset_features(df.copy(), window=5, max_pairs=5, reduce="top_k")
    limited_time = time.perf_counter() - start

    start = time.perf_counter()
    full = add_cross_asset_features(df.copy(), window=5, max_pairs=None)
    full_time = time.perf_counter() - start

    pair_cols_limited = [
        c for c in limited.columns if c.startswith("corr_") or c.startswith("relret_")
    ]
    pair_cols_full = [
        c for c in full.columns if c.startswith("corr_") or c.startswith("relret_")
    ]
    assert len(pair_cols_limited) == 5 * 4
    assert len(pair_cols_limited) < len(pair_cols_full)
    assert limited_time < full_time


def test_cross_asset_default_pca_reduces_columns_large_universe():
    symbols = [f"S{i:03d}" for i in range(40)]
    periods = 20
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "Timestamp": idx.repeat(len(symbols)),
            "Symbol": symbols * periods,
            "return": rng.standard_normal(periods * len(symbols)) / 100,
        }
    )

    full = add_cross_asset_features(df.copy(), window=5, max_pairs=None)
    pca_df = add_cross_asset_features(df.copy(), window=5, max_pairs=5)

    pca_cols = [c for c in pca_df.columns if c.startswith("pair_pca_")]
    full_cols = [
        c for c in full.columns if c.startswith("corr_") or c.startswith("relret_")
    ]
    assert len(pca_cols) == 5
    assert len(pca_cols) < len(full_cols)
    assert not any(
        c.startswith("corr_") or c.startswith("relret_") for c in pca_df.columns
    )


def test_cross_asset_warns_when_limit_exceeded(caplog):
    symbols = [f"S{i:03d}" for i in range(25)]
    periods = 20
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "Timestamp": idx.repeat(len(symbols)),
            "Symbol": symbols * periods,
            "return": rng.standard_normal(periods * len(symbols)) / 100,
        }
    )

    with caplog.at_level("WARNING", logger=cross_asset.logger.name):
        limited = add_cross_asset_features(df.copy(), window=5)

    assert any("whitelist" in record.getMessage() for record in caplog.records)
    assert any(col.startswith("pair_pca_") for col in limited.columns)
