"""Cross asset and market wide features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - polars optional
    import polars as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:  # pragma: no cover - decorator optional in standalone tests
    from . import validate_module
except Exception:  # pragma: no cover - fallback when imported directly

    def validate_module(func):
        return func
logger = logging.getLogger(__name__)


DEFAULT_MAX_PAIRS = 200


def _relative_strength_features(pivot: pd.DataFrame) -> pd.DataFrame:
    """Return relative strength features aligned to ``(Timestamp, Symbol)``.

    The returned dataframe has a ``MultiIndex`` of ``Timestamp`` and ``Symbol``
    and columns named ``rel_strength_<symbol>``.  Missing values are preserved
    so that the caller can decide on the appropriate fill strategy.
    """

    if pivot.empty:
        return pd.DataFrame()

    cs_mean = pivot.mean(axis=1)
    cs_std = pivot.std(axis=1, ddof=0).replace(0, np.nan)
    rel_strength = pivot.sub(cs_mean, axis=0).div(cs_std, axis=0)

    rel_strength_long = (
        rel_strength.stack(dropna=False)
        .rename("value")
        .reset_index()
        .rename(columns={"Symbol": "feature_symbol"})
    )
    if rel_strength_long.empty:
        return pd.DataFrame()

    rel_strength_long["feature"] = (
        "rel_strength_" + rel_strength_long["feature_symbol"].astype(str)
    )
    rel_strength_wide = rel_strength_long.pivot(
        index=["Timestamp", "feature_symbol"],
        columns="feature",
        values="value",
    )
    rel_strength_wide.index = rel_strength_wide.index.rename([
        "Timestamp",
        "Symbol",
    ])
    return rel_strength_wide


def _pairwise_features(
    pair_pivot: pd.DataFrame, window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling correlation and relative return features.

    Parameters
    ----------
    pair_pivot:
        Wide dataframe with ``Timestamp`` index and columns for each symbol's
        return series restricted to the requested universe.
    window:
        Lookback window for the rolling correlation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The first element contains pairwise features indexed by
        ``(Timestamp, Symbol)`` with columns for ``corr_*`` and ``relret_*``
        interactions.  The second element is the long format correlation table
        used for constructing ``cross_confirm`` columns.
    """

    if pair_pivot.empty:
        return pd.DataFrame(), pd.DataFrame()

    corr = pair_pivot.rolling(window).corr()
    corr = corr.rename_axis(index=["Timestamp", "Symbol"], columns="other")
    corr_long = (
        corr.stack(dropna=False)
        .rename("value")
        .reset_index()
    )
    if corr_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    corr_long = corr_long[corr_long["Symbol"] != corr_long["other"]]
    corr_features = corr_long.assign(
        feature=lambda d: "corr_" + d["Symbol"].astype(str) + "_" + d["other"].astype(str)
    ).pivot(index=["Timestamp", "Symbol"], columns="feature", values="value")

    pair_symbols = list(pair_pivot.columns)
    arr = pair_pivot.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        relret_arr = arr[:, :, None] / arr[:, None, :]
    relret_arr[~np.isfinite(relret_arr)] = np.nan
    relret_index = pd.MultiIndex.from_product(
        [pair_pivot.index, pair_symbols], names=["Timestamp", "Symbol"]
    )
    relret_df = pd.DataFrame(
        relret_arr.reshape(len(pair_pivot.index) * len(pair_symbols), len(pair_symbols)),
        index=relret_index,
        columns=pair_symbols,
    )
    relret_long = (
        relret_df.stack(dropna=False)
        .rename("value")
        .reset_index()
        .rename(columns={"level_2": "other"})
    )
    relret_long = relret_long[relret_long["Symbol"] != relret_long["other"]]
    relret_features = relret_long.assign(
        feature=lambda d: "relret_"
        + d["Symbol"].astype(str)
        + "_"
        + d["other"].astype(str)
    ).pivot(index=["Timestamp", "Symbol"], columns="feature", values="value")

    pair_features = pd.concat([corr_features, relret_features], axis=1)
    return pair_features, corr_long


try:  # pragma: no cover - validators optional when running in isolation
    from .validators import require_columns, assert_no_nan
except Exception:  # pragma: no cover - graceful fallback

    def require_columns(df, cols, **_):  # type: ignore[unused-arg]
        return df

    def assert_no_nan(df, cols=None, **_):  # type: ignore[unused-arg]
        return df


def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pd.merge_asof(left, right, **kwargs)


def load_index_ohlc(source: str):
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


def add_index_features(df):
    try:
        from utils import load_config

        cfg = load_config()
        index_sources = cfg.get("index_data", {})
    except Exception:  # pragma: no cover - config issues shouldn't fail
        index_sources = {}

    if pl is not None and isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
        pdf = add_index_features(pdf)
        return pl.from_pandas(pdf)
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
    df,
    window: int = 30,
    whitelist: Iterable[str] | None = None,
    max_pairs: int | None = DEFAULT_MAX_PAIRS,
    reduce: str = "pca",
    confirm_peers: Iterable[str] | None = None,
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

    When the requested symbol universe would generate more than ``max_pairs``
    unique pairs a warning is emitted and the chosen reduction strategy is
    applied automatically (``reduce='pca'`` by default). Provide ``max_pairs=None``
    to opt-out of the limit entirely.

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
        Maximum number of pairwise relationships to keep (defaults to
        :data:`DEFAULT_MAX_PAIRS`).  Set to ``None`` to disable the limit and
        generate the full feature set.
    reduce:
        Reduction strategy when ``max_pairs`` is set. Options are ``"top_k"``
        or ``"pca"`` (default) for dimensionality compression.
    confirm_peers:
        Optional iterable of peer symbols for which ``cross_confirm``
        correlation columns will be exposed.  For each requested peer a
        ``cross_confirm_<peer>`` column is created containing the rolling
        correlation between the current symbol and that peer.  Missing values
        are filled with ``0.0``.
    """

    if pl is not None and isinstance(df, pl.DataFrame):
        pdf = add_cross_asset_features(
            df.to_pandas(),
            window=window,
            whitelist=whitelist,
            max_pairs=max_pairs,
            reduce=reduce,
            confirm_peers=confirm_peers,
        )
        return pl.from_pandas(pdf)

    required = {"Symbol", "Timestamp", "return"}
    if not required.issubset(df.columns):
        return df

    df = df.copy().sort_values("Timestamp")
    original_attrs = dict(df.attrs)
    pivot = df.pivot(index="Timestamp", columns="Symbol", values="return")
    symbols = list(pivot.columns)

    df_indexed = df.set_index(["Timestamp", "Symbol"])
    new_columns: list[str] = []
    new_column_set: set[str] = set()

    def _register_columns(cols: Iterable[str]) -> None:
        for col in cols:
            if col not in new_column_set:
                new_column_set.add(col)
                new_columns.append(col)

    rel_strength_df = _relative_strength_features(pivot)
    meta = {
        "relative_strength": 0,
        "pair_features": 0,
        "confirm_peers": 0,
    }
    if not rel_strength_df.empty:
        df_indexed = df_indexed.join(rel_strength_df, how="left")
        rel_cols = list(rel_strength_df.columns)
        meta["relative_strength"] = len(rel_cols)
        _register_columns(rel_cols)

    if reduce not in {"top_k", "pca"}:
        raise ValueError("reduce must be one of 'top_k', 'pca'")

    pair_symbols = [s for s in symbols if whitelist is None or s in whitelist]
    pair_symbol_count = len(pair_symbols)
    unique_pairs = pair_symbol_count * (pair_symbol_count - 1) // 2
    should_reduce = (
        max_pairs is not None and max_pairs > 0 and unique_pairs > max_pairs
    )

    reduction_mode = "none"
    if pair_symbol_count >= 2:
        pair_pivot = pivot[pair_symbols]
        pair_features, corr_long = _pairwise_features(pair_pivot, window)

        reduction_mode = "full"
        if should_reduce:
            reduction_mode = reduce
            logger.warning(
                (
                    "Requested %s symbol pairs for cross-asset features exceeds "
                    "max_pairs=%s. %s reduction will be applied; provide a whitelist "
                    "to target specific symbols."
                ),
                unique_pairs,
                max_pairs,
                "PCA" if reduce == "pca" else "Top-k",
            )

        if not pair_features.empty:
            if reduce == "top_k" and should_reduce and max_pairs:
                corr_matrix = pair_pivot.corr().abs().fillna(0.0)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                top_pairs = (
                    corr_matrix.where(mask)
                    .stack()
                    .sort_values(ascending=False)
                    .head(max_pairs)
                    .index.tolist()
                )
                keep_cols: list[str] = []
                seen: set[str] = set()
                for s1, s2 in top_pairs:
                    for col in (
                        f"corr_{s1}_{s2}",
                        f"relret_{s1}_{s2}",
                        f"corr_{s2}_{s1}",
                        f"relret_{s2}_{s1}",
                    ):
                        if col in pair_features.columns and col not in seen:
                            keep_cols.append(col)
                            seen.add(col)
                if keep_cols:
                    subset = pair_features[keep_cols]
                    df_indexed = df_indexed.join(subset, how="left")
                    meta["pair_features"] = len(keep_cols)
                    _register_columns(keep_cols)
                else:
                    meta["pair_features"] = 0
            elif reduce == "pca" and should_reduce and max_pairs:
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
                df_indexed = df_indexed.join(reduced_df, how="left")
                meta["pair_features"] = len(reduced_cols)
                _register_columns(reduced_cols)
            else:
                df_indexed = df_indexed.join(pair_features, how="left")
                meta["pair_features"] = pair_features.shape[1]
                _register_columns(pair_features.columns.tolist())
        else:
            corr_long = pd.DataFrame(columns=["Timestamp", "Symbol", "other", "value"])
    else:
        corr_long = pd.DataFrame(columns=["Timestamp", "Symbol", "other", "value"])

    if confirm_peers:
        peers = [p for p in confirm_peers if p in symbols]
        confirm_cols: list[str] = []
        if peers and not corr_long.empty:
            confirm_df = (
                corr_long[corr_long["other"].isin(peers)]
                .assign(feature=lambda d: "cross_confirm_" + d["other"].astype(str))
                .pivot(index=["Timestamp", "Symbol"], columns="feature", values="value")
            )
            if not confirm_df.empty:
                df_indexed = df_indexed.join(confirm_df, how="left")
                confirm_cols = confirm_df.columns.tolist()
                _register_columns(confirm_cols)
        for peer in peers:
            colname = f"cross_confirm_{peer}"
            if colname not in df_indexed.columns:
                df_indexed[colname] = 0.0
                confirm_cols.append(colname)
                _register_columns([colname])
        if peers:
            meta["confirm_peers"] = len({f"cross_confirm_{p}" for p in peers})

    if new_columns:
        df_indexed[new_columns] = df_indexed[new_columns].fillna(0.0)

    result = df_indexed.reset_index()
    result.index = df.index
    result.attrs.update(original_attrs)
    result.attrs["cross_asset_feature_summary"] = {
        "symbol_count": len(symbols),
        "unique_pairs": unique_pairs,
        "relative_strength_columns": meta["relative_strength"],
        "pair_columns": meta["pair_features"],
        "confirm_columns": meta["confirm_peers"],
        "reduction": reduction_mode,
        "max_pairs": max_pairs,
    }
    return result


@validate_module
def compute(df) -> pd.DataFrame:
    """Enrich ``df`` with index and cross-asset features.

    The following columns are appended:

    - Market index returns/volatility via :func:`add_index_features`.
    - Pairwise rolling correlations (``corr_<sym1>_<sym2>``) and relative
      return ratios (``relret_<sym1>_<sym2>``) via
      :func:`add_cross_asset_features`.

    Additionally, rolling adjacency matrices describing the cross-symbol
    relationships are stored in ``df.attrs['adjacency_matrices']``.
    """

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
        k: cfg[k]
        for k in ["window", "whitelist", "max_pairs", "reduce", "confirm_peers"]
        if k in cfg
    }
    df = add_cross_asset_features(df, **params)

    if {"Symbol", "Timestamp"}.issubset(df.columns):
        try:
            window = int(cfg.get("adjacency_window", 30))
        except Exception:  # pragma: no cover
            window = 30
        try:
            pivot = (
                df.pivot(index="Timestamp", columns="Symbol", values="return")
                .sort_index()
                .fillna(0.0)
            )
            matrices = {}
            for end in pivot.index[window - 1 :]:
                window_df = pivot.loc[:end].tail(window)
                mat = (
                    window_df.corr().fillna(0.0).to_numpy(dtype=np.float32)
                )
                np.fill_diagonal(mat, 0.0)
                matrices[end] = mat
        except Exception:  # pragma: no cover - fallback to identity
            symbols = sorted(df["Symbol"].unique())
            eye = np.eye(len(symbols), dtype=np.float32)
            matrices = {ts: eye for ts in pd.to_datetime(df["Timestamp"]).unique()}
        df.attrs["adjacency_matrices"] = [
            (ts, matrices[ts]) for ts in sorted(matrices.keys())
        ]
    return df


add_index_features.supports_polars = True  # type: ignore[attr-defined]
add_cross_asset_features.supports_polars = True  # type: ignore[attr-defined]
compute.supports_polars = True  # type: ignore[attr-defined]


__all__ = [
    "DEFAULT_MAX_PAIRS",
    "add_index_features",
    "add_cross_asset_features",
    "compute",
]
