"""Cross asset and market wide features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - decorator optional in standalone tests
    from . import validate_module
except Exception:  # pragma: no cover - fallback when imported directly

    def validate_module(func):
        return func


try:  # pragma: no cover - validators optional when running in isolation
    from .validators import require_columns, assert_no_nan
except Exception:  # pragma: no cover - graceful fallback

    def require_columns(df, cols, **_):  # type: ignore[unused-arg]
        return df

    def assert_no_nan(df, cols=None, **_):  # type: ignore[unused-arg]
        return df


def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pd.merge_asof(left, right, **kwargs)


def load_index_ohlc(source: str) -> pd.DataFrame:
    try:
        if str(source).startswith("http"):
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(Path(source))
    except Exception:
        return pd.DataFrame()
    if "Date" not in df.columns:
        df.columns = [c.strip() for c in df.columns]
    return df


def add_index_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from utils import load_config

        cfg = load_config()
        index_sources = cfg.get("index_data", {})
    except Exception:  # pragma: no cover - config issues shouldn't fail
        index_sources = {}

    df = df.sort_values("Timestamp")
    for name, src in index_sources.items():
        idx_df = load_index_ohlc(src)
        if {"Date", "Close"}.issubset(idx_df.columns):
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            idx_df = idx_df.sort_values("Date")
            idx_df["return"] = idx_df["Close"].pct_change()
            idx_df["vol"] = idx_df["return"].rolling(21).std()
            idx_df = idx_df[["Date", "return", "vol"]]
            idx_df = idx_df.rename(
                columns={
                    "Date": "index_date",
                    "return": f"{name}_ret",
                    "vol": f"{name}_vol",
                }
            )
            df = _merge_asof(
                df,
                idx_df,
                left_on="Timestamp",
                right_on="index_date",
                direction="backward",
            ).drop(columns=["index_date"])
        else:
            df[f"{name}_ret"] = 0.0
            df[f"{name}_vol"] = 0.0

    if not index_sources:
        for col in ["sp500_ret", "sp500_vol", "vix_ret", "vix_vol"]:
            if col not in df.columns:
                df[col] = 0.0
    return df


def add_cross_asset_features(
    df: pd.DataFrame,
    window: int = 30,
    whitelist: Iterable[str] | None = None,
    max_pairs: int | None = None,
    reduce: str = "pca",
) -> pd.DataFrame:
    """Add simple cross-asset interaction features.

    For every pair of symbols sharing the same ``Timestamp`` this function
    computes pairwise interactions.  By default all symbol combinations are
    used, however ``max_pairs`` together with ``reduce`` can limit or compress
    the generated features:

    - ``reduce='top_k'`` keeps only the ``max_pairs`` most correlated symbol
      pairs (unique, unordered) and skips the rest.
    - ``reduce='pca'`` (default) applies PCA to the full pairwise feature
      matrix and returns ``max_pairs`` principal components labelled
      ``pair_pca_<i>``.

    Missing values (for example during the warm up period of the rolling
    correlation) are filled with ``0.0`` to keep downstream models simple.

    Parameters
    ----------
    df:
        Input data containing at least ``Symbol``, ``Timestamp`` and ``return``.
    window:
        Lookback window for the rolling correlation.
    whitelist:
        Optional iterable restricting pairwise calculations to the provided
        symbols.  If ``None`` all symbols are considered.
    max_pairs:
        Maximum number of pairwise relationships to keep.  If ``None`` all
        combinations are generated.
    reduce:
        Reduction strategy when ``max_pairs`` is set. Options are ``"top_k"``
        or ``"pca"`` (default) for dimensionality compression.
    """

    required = {"Symbol", "Timestamp", "return"}
    if not required.issubset(df.columns):
        return df

    df = df.copy().sort_values("Timestamp")
    pivot = df.pivot(index="Timestamp", columns="Symbol", values="return")
    symbols = list(pivot.columns)

    # ------------------------------------------------------------------
    # Cross-sectional relative strength
    # ------------------------------------------------------------------
    cs_mean = pivot.mean(axis=1)
    cs_std = pivot.std(axis=1, ddof=0).replace(0, np.nan)
    rel_strength = pivot.sub(cs_mean, axis=0).div(cs_std, axis=0)

    for sym in symbols:
        ts_map = df.loc[df["Symbol"] == sym, "Timestamp"]
        df.loc[df["Symbol"] == sym, f"rel_strength_{sym}"] = ts_map.map(
            rel_strength[sym]
        ).fillna(0.0)
    rel_cols = [f"rel_strength_{sym}" for sym in symbols]
    df[rel_cols] = df[rel_cols].fillna(0.0)

    # ------------------------------------------------------------------
    # Pairwise interactions
    # ------------------------------------------------------------------
    if reduce not in {"top_k", "pca"}:
        raise ValueError("reduce must be one of 'top_k', 'pca'")

    pair_symbols = [s for s in symbols if whitelist is None or s in whitelist]
    if len(pair_symbols) >= 2:
        pair_pivot = pivot[pair_symbols]

        if max_pairs and reduce == "top_k":
            # ----------------------------------------------------------
            # Identify top-k correlated symbol pairs
            # ----------------------------------------------------------
            corr_matrix = pair_pivot.corr().abs().fillna(0.0)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            top_pairs = (
                corr_matrix.where(mask)
                .stack()
                .sort_values(ascending=False)
                .head(max_pairs)
                .index.tolist()
            )

            for s1, s2 in top_pairs:
                corr_series = pair_pivot[s1].rolling(window).corr(pair_pivot[s2])
                ratio = (pair_pivot[s1] / pair_pivot[s2]).replace(
                    [np.inf, -np.inf], np.nan
                )
                inv_ratio = (pair_pivot[s2] / pair_pivot[s1]).replace(
                    [np.inf, -np.inf], np.nan
                )

                ts_map = df.loc[df["Symbol"] == s1, "Timestamp"]
                df.loc[df["Symbol"] == s1, f"corr_{s1}_{s2}"] = ts_map.map(corr_series)
                df.loc[df["Symbol"] == s1, f"relret_{s1}_{s2}"] = ts_map.map(ratio)

                ts_map = df.loc[df["Symbol"] == s2, "Timestamp"]
                df.loc[df["Symbol"] == s2, f"corr_{s2}_{s1}"] = ts_map.map(corr_series)
                df.loc[df["Symbol"] == s2, f"relret_{s2}_{s1}"] = ts_map.map(inv_ratio)

            keep_cols: list[str] = []
            for s1, s2 in top_pairs:
                keep_cols.extend(
                    [
                        f"corr_{s1}_{s2}",
                        f"relret_{s1}_{s2}",
                        f"corr_{s2}_{s1}",
                        f"relret_{s2}_{s1}",
                    ]
                )
            df[keep_cols] = df[keep_cols].fillna(0.0)

        else:
            # ----------------------------------------------------------
            # Full pairwise matrices (possibly reduced via PCA)
            # ----------------------------------------------------------
            corr = pair_pivot.rolling(window).corr()
            corr = corr.rename_axis(index=["Timestamp", "Symbol"], columns="other")
            corr_long = corr.stack().reset_index().rename(columns={0: "value"})
            corr_long = corr_long[corr_long["Symbol"] != corr_long["other"]]
            corr_features = corr_long.assign(
                feature=lambda d: "corr_" + d["Symbol"] + "_" + d["other"]
            ).pivot(index=["Timestamp", "Symbol"], columns="feature", values="value")

            arr = pair_pivot.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                relret_arr = arr[:, :, None] / arr[:, None, :]
            relret_arr[~np.isfinite(relret_arr)] = np.nan
            relret_df = pd.DataFrame(
                relret_arr.reshape(
                    len(pair_pivot.index) * len(pair_symbols), len(pair_symbols)
                ),
                index=pd.MultiIndex.from_product(
                    [pair_pivot.index, pair_symbols], names=["Timestamp", "Symbol"]
                ),
                columns=pair_symbols,
            )
            relret_long = (
                relret_df.stack()
                .reset_index()
                .rename(columns={"level_2": "other", 0: "value"})
            )
            relret_long = relret_long[relret_long["Symbol"] != relret_long["other"]]
            relret_features = relret_long.assign(
                feature=lambda d: "relret_" + d["Symbol"] + "_" + d["other"]
            ).pivot(index=["Timestamp", "Symbol"], columns="feature", values="value")

            pair_features = pd.concat([corr_features, relret_features], axis=1)

            if max_pairs and reduce == "pca":
                X = pair_features.fillna(0.0).to_numpy()
                n_comp = min(max_pairs, X.shape[1])
                try:  # pragma: no cover - sklearn optional
                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=n_comp)
                    reduced = pca.fit_transform(X)
                except Exception:  # pragma: no cover - fallback without sklearn
                    U, S, Vt = np.linalg.svd(X, full_matrices=False)
                    reduced = U[:, :n_comp] * S[:n_comp]
                reduced_cols = [f"pair_pca_{i}" for i in range(reduced.shape[1])]
                reduced_df = pd.DataFrame(
                    reduced, index=pair_features.index, columns=reduced_cols
                )
                df = df.join(reduced_df, on=["Timestamp", "Symbol"])
                df[reduced_cols] = df[reduced_cols].fillna(0.0)
            else:
                df = df.join(pair_features, on=["Timestamp", "Symbol"])
                df[pair_features.columns] = df[pair_features.columns].fillna(0.0)

    return df


@validate_module
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich ``df`` with index and cross-asset features.

    The following columns are appended:

    - Market index returns/volatility via :func:`add_index_features`.
    - Pairwise rolling correlations (``corr_<sym1>_<sym2>``) and relative
      return ratios (``relret_<sym1>_<sym2>``) via
      :func:`add_cross_asset_features`.

    Additionally, rolling adjacency matrices describing the cross-symbol
    relationships are stored in ``df.attrs['adjacency_matrices']``.
    """

    from data.graph_builder import build_rolling_adjacency

    try:  # pragma: no cover - config optional in tests
        from utils import load_config

        cfg = load_config().get("cross_asset", {})
    except Exception:  # pragma: no cover - config issues shouldn't fail
        cfg = {}

    # Validate essential inputs before heavy processing
    require_columns(df, ["Timestamp", "Symbol", "return"])
    assert_no_nan(df, ["return"])

    df = add_index_features(df)
    params = {
        k: cfg[k] for k in ["window", "whitelist", "max_pairs", "reduce"] if k in cfg
    }
    df = add_cross_asset_features(df, **params)

    if {"Symbol", "Timestamp"}.issubset(df.columns):
        try:
            matrices = build_rolling_adjacency(df)
        except Exception:  # pragma: no cover - fallback to identity
            symbols = sorted(df["Symbol"].unique())
            eye = np.eye(len(symbols))
            matrices = {ts: eye for ts in pd.to_datetime(df["Timestamp"]).unique()}
        df.attrs["adjacency_matrices"] = matrices
    return df


__all__ = ["add_index_features", "add_cross_asset_features", "compute"]
