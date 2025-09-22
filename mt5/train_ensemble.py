"""Train multiple models and build an ensemble."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import importlib.util
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from analysis.ensemble_diversity import error_correlation_matrix
from analysis.information_coefficient import (
    information_coefficient,
    information_coefficient_series,
)


# Lightweight import of mixture-of-experts utilities without requiring the full
# :mod:`models` package (which pulls in heavy optional dependencies).  This
# mirrors the approach used in ``models/mixure_of_experts.py`` itself.
_moe_spec = importlib.util.spec_from_file_location(
    "mixture_of_experts", Path(__file__).resolve().parent / "models" / "mixture_of_experts.py"
)
_moe = importlib.util.module_from_spec(_moe_spec)
assert _moe_spec and _moe_spec.loader
sys.modules["mixture_of_experts"] = _moe
_moe_spec.loader.exec_module(_moe)  # type: ignore

TrendExpert = _moe.TrendExpert
MeanReversionExpert = _moe.MeanReversionExpert
MacroExpert = _moe.MacroExpert
ExpertSpec = _moe.ExpertSpec
GatingNetwork = _moe.GatingNetwork
ResourceCapabilities = _moe.ResourceCapabilities


def build_moe(
    cfg: dict | None = None,
) -> tuple[GatingNetwork, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Construct base experts and a gating network.

    Parameters
    ----------
    cfg:
        Optional configuration providing ``expert_weights`` (list of floats),
        ``diversity_weights`` and ``risk_budgets``. Sharpness controls the
        softness of the gating network's selection function and defaults to
        ``5.0`` when unspecified.
    """

    cfg = cfg or {}
    sharpness = float(cfg.get("sharpness", 5.0))
    expert_weights = np.asarray(cfg.get("expert_weights", [1.0, 1.0, 1.0]), dtype=float)
    diversity_cfg = cfg.get("diversity_weights")
    div_arr = (
        np.asarray(diversity_cfg, dtype=float) if diversity_cfg is not None else None
    )

    experts = [
        ExpertSpec(TrendExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
        ExpertSpec(MeanReversionExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
        ExpertSpec(MacroExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
    ]

    risk_cfg = cfg.get("risk_budgets")
    risk_arr: np.ndarray | None
    if risk_cfg is None:
        risk_arr = None
    else:
        n_exp = len(experts)
        if isinstance(risk_cfg, dict):
            lowered = {str(k).lower(): float(v) for k, v in risk_cfg.items()}
            vals = []
            for spec in experts:
                cls_name = spec.model.__class__.__name__
                candidates = [
                    cls_name,
                    cls_name.lower(),
                    cls_name.replace("Expert", ""),
                    cls_name.replace("Expert", "").lower(),
                ]
                value = None
                for key in candidates:
                    val = lowered.get(str(key).lower())
                    if val is not None:
                        value = val
                        break
                if value is None:
                    raise ValueError(
                        f"Missing risk budget for expert '{cls_name}'."
                    )
                vals.append(float(value))
            risk_arr = np.asarray(vals, dtype=float)
        else:
            risk_arr = np.asarray(risk_cfg, dtype=float)
            if risk_arr.shape != (len(experts),):
                raise ValueError(
                    "risk_budgets must provide one entry per expert"
                )
        risk_arr = np.nan_to_num(risk_arr, nan=0.0, posinf=0.0, neginf=0.0)
        risk_arr = np.clip(risk_arr, 0.0, None)
        total_budget = risk_arr.sum()
        if total_budget <= 0.0 or not np.isfinite(total_budget):
            risk_arr = None
        else:
            risk_arr = risk_arr / total_budget

    return GatingNetwork(experts, sharpness=sharpness), expert_weights, div_arr, risk_arr


def _normalise_weights(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(arr) / len(arr)
    return arr / total


def _combine_preferences(
    base: np.ndarray,
    preference: np.ndarray | None,
    risk_budgets: np.ndarray | None,
) -> np.ndarray:
    weights = np.asarray(base, dtype=float)
    if preference is not None:
        pref = np.asarray(preference, dtype=float)
        if pref.shape != weights.shape:
            raise ValueError("expert weight size mismatch")
        pref = np.nan_to_num(pref, nan=0.0, posinf=0.0, neginf=0.0)
        pref = np.clip(pref, 0.0, None)
        weights = weights * pref
    if risk_budgets is not None:
        return GatingNetwork._apply_budgets(weights, risk_budgets)
    return _normalise_weights(weights)


def _inverse_correlation_weights(corr: pd.DataFrame) -> np.ndarray:
    if corr.empty:
        return np.array([])
    corr_abs = corr.abs().to_numpy()
    np.fill_diagonal(corr_abs, np.nan)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(corr_abs, axis=1)
    counts = np.sum(np.isfinite(corr_abs), axis=1)
    mean_corr = np.zeros_like(sums, dtype=float)
    mask = counts > 0
    mean_corr[mask] = sums[mask] / counts[mask]
    mean_corr = np.nan_to_num(mean_corr, nan=0.0, posinf=0.0, neginf=0.0)
    inv = 1.0 / (1e-6 + mean_corr)
    inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
    inv = np.clip(inv, 0.0, None)
    if inv.sum() <= 0.0 or not np.isfinite(inv.sum()):
        inv = np.ones_like(inv)
    return _normalise_weights(inv)


def _combine_diversity(
    primary: np.ndarray | None,
    secondary: np.ndarray | None,
) -> np.ndarray | None:
    if primary is None and secondary is None:
        return None
    if primary is None:
        arr = np.asarray(secondary, dtype=float)
    elif secondary is None:
        arr = np.asarray(primary, dtype=float)
    else:
        arr = np.asarray(primary, dtype=float) * np.asarray(secondary, dtype=float)
    return _normalise_weights(arr)


def predict_mixture(
    history: Sequence[float],
    market_regime: float,
    caps: ResourceCapabilities,
    cfg: dict | None = None,
) -> float:
    """Return the gated mixture prediction for ``history``.

    The function builds the experts and gating network using :func:`build_moe`
    and then combines their predictions according to the current market regime
    and available resource capabilities.
    """

    gating, expert_weights, diversity, risk_budgets = build_moe(cfg)
    base = gating.weights(market_regime, caps, diversity, risk_budgets=risk_budgets)
    weights = _combine_preferences(base, expert_weights, risk_budgets)
    preds = np.array([spec.model.predict(history) for spec in gating.experts])
    return float(np.dot(weights, preds))


def train_moe_ensemble(
    histories: Sequence[Sequence[float]],
    regimes: Sequence[float],
    targets: Sequence[float],
    caps: ResourceCapabilities,
    cfg: dict | None = None,
    return_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train mixture-of-experts on validation data and log metrics.

    Parameters
    ----------
    histories:
        Sequence of historical price windows.
    regimes:
        Validation regime labels aligned with ``histories``.
    targets:
        True next-step values for each history window.
    caps:
        Available resource capabilities.
    cfg:
        Optional configuration passed to :func:`build_moe`.
    return_weights:
        When ``True`` an additional array containing the per-sample mixture
        weights is returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The mixture predictions and a ``(n_samples, n_experts)`` array of expert
        predictions. When ``return_weights`` is ``True`` a third array containing
        the normalised mixture weights for each sample is returned.
    """

    try:  # pragma: no cover - optional dependency
        from analytics import mlflow_client as mlflow
        if not hasattr(mlflow, "log_dict"):  # provide passthrough if missing
            import mlflow as _mlf  # type: ignore

            mlflow.log_dict = lambda data, name: _mlf.log_dict(data, name)
    except Exception:  # noqa: BLE001 - lightweight stub used in tests
        import contextlib
        import types

        mlflow = types.SimpleNamespace(
            start_run=lambda *a, **k: contextlib.nullcontext(),
            log_dict=lambda *a, **k: None,
            log_metric=lambda *a, **k: None,
            end_run=lambda *a, **k: None,
        )

    gating, expert_weights, div_cfg, risk_budgets = build_moe(cfg)
    preds = np.array(
        [[spec.model.predict(h) for spec in gating.experts] for h in histories]
    )
    truth = np.asarray(targets, dtype=float)
    regimes_arr = np.asarray(regimes)
    val_preds = {f"exp_{i}": preds[:, i] for i in range(preds.shape[1])}
    corr = error_correlation_matrix(val_preds, truth)
    global_div = _inverse_correlation_weights(corr) if not corr.empty else None
    base_diversity = _combine_diversity(global_div, div_cfg)

    regime_diversity: dict[float | int | str, np.ndarray] = {}
    unique_regimes = np.unique(regimes_arr)
    for reg in unique_regimes:
        if pd.isna(reg):
            continue
        mask = regimes_arr == reg
        if np.count_nonzero(mask) < 2:
            continue
        reg_preds = {f"exp_{i}": preds[mask, i] for i in range(preds.shape[1])}
        reg_corr = error_correlation_matrix(reg_preds, truth[mask])
        reg_div = _inverse_correlation_weights(reg_corr)
        combined = _combine_diversity(reg_div, div_cfg)
        if combined is not None:
            regime_diversity[reg] = combined
    if base_diversity is not None:
        regime_diversity.setdefault("default", base_diversity)
    diversity_input: dict[float | int | str, np.ndarray] | np.ndarray | None
    if regime_diversity:
        diversity_input = regime_diversity
    else:
        diversity_input = base_diversity

    ic_series = information_coefficient_series(val_preds, truth)

    mlflow.start_run("moe_training", cfg or {})
    try:
        if hasattr(mlflow, "log_dict"):
            mlflow.log_dict(corr.to_dict(), "error_correlation.json")
            mlflow.log_dict(
                {f"expert_{i}": float(ic_series.iloc[i]) for i in range(len(ic_series))},
                "moe_information_coefficients.json",
            )
            if isinstance(diversity_input, dict):
                mlflow.log_dict(
                    {
                        str(key): [float(x) for x in arr]
                        for key, arr in diversity_input.items()
                    },
                    "regime_diversity_weights.json",
                )
            elif diversity_input is not None:
                mlflow.log_dict(
                    {f"w_{i}": float(w) for i, w in enumerate(diversity_input)},
                    "gating_weights.json",
                )
            if risk_budgets is not None:
                mlflow.log_dict(
                    {f"budget_{i}": float(b) for i, b in enumerate(risk_budgets)},
                    "risk_budgets.json",
                )

        mix_preds: list[float] = []
        weight_history: list[np.ndarray] = [] if return_weights else []
        for reg, expert_pred in zip(regimes_arr, preds):
            base = gating.weights(
                float(reg),
                caps,
                diversity_input,
                risk_budgets=risk_budgets,
            )
            combined = _combine_preferences(base, expert_weights, risk_budgets)
            mix_preds.append(float(np.dot(combined, expert_pred)))
            if return_weights:
                weight_history.append(combined)
        mix_arr = np.asarray(mix_preds)
        mse_mix = float(np.mean((mix_arr - truth) ** 2))
        mse_experts = [
            float(np.mean((preds[:, i] - truth) ** 2))
            for i in range(preds.shape[1])
        ]
        corr_abs = corr.abs().values
        if corr_abs.size:
            np.fill_diagonal(corr_abs, np.nan)
            with np.errstate(all="ignore"):
                mean_corr_val = np.nanmean(corr_abs)
            mlflow.log_metric(
                "mean_error_correlation",
                float(np.nan_to_num(mean_corr_val, nan=0.0)),
            )
        mlflow.log_metric("mse_mixture", mse_mix)
        for i, m in enumerate(mse_experts):
            mlflow.log_metric(f"mse_expert_{i}", m)
        for i, ic in enumerate(ic_series):
            if np.isfinite(ic):
                mlflow.log_metric(f"ic_expert_{i}", float(ic))
        mlflow.log_metric("mixture_improvement", min(mse_experts) - mse_mix)
    finally:
        mlflow.end_run()

    if return_weights:
        return mix_arr, preds, np.asarray(weight_history)
    return mix_arr, preds

logger = logging.getLogger(__name__)


@dataclass
class _WrappedModel:
    model: Any
    features: Iterable[str] | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.features is not None:
            X = X[list(self.features)]
        return self.model.predict_proba(X)


def _train_lightgbm(X: pd.DataFrame, y: np.ndarray, params: Mapping[str, Any] | None) -> Any:
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(**(params or {}))
    clf.fit(X, y)
    return clf


def _train_cross_asset_transformer(
    X: pd.DataFrame, y: np.ndarray, params: Mapping[str, Any] | None
) -> Any:
    try:  # pragma: no cover - optional dependency
        from models.cross_asset_transformer import CrossAssetTransformer
    except Exception as e:  # noqa: BLE001
        raise ImportError("CrossAssetTransformer requires PyTorch") from e
    import torch

    epochs = int(params.get("epochs", 10)) if params else 10
    net = CrossAssetTransformer(
        input_dim=X.shape[1], n_symbols=1, output_dim=1, **{k: v for k, v in (params or {}).items() if k != "epochs"}
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    X_t = torch.tensor(X.values, dtype=torch.float32).view(-1, 1, 1, X.shape[1])
    times_t = torch.zeros(X_t.shape[0], 1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = net(X_t, times_t).view(-1, 1)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()
    net.eval()

    class _Wrapper:
        def __init__(self, net):
            self.net = net

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            with torch.no_grad():
                x = torch.tensor(X.values, dtype=torch.float32).view(-1, 1, 1, X.shape[1])
                times = torch.zeros(x.shape[0], 1, 1)
                logits = self.net(x, times).view(-1)
                prob = torch.sigmoid(logits).numpy()
            return np.column_stack([1 - prob, prob])

    return _Wrapper(net)


def _train_neural_quantile(
    X: pd.DataFrame, y: np.ndarray, params: Mapping[str, Any] | None
) -> Any:
    from models.quantile_regression import NeuralQuantile

    nq = NeuralQuantile(input_dim=X.shape[1], alphas=[0.5], **(params or {}))
    nq.fit(X.values, y)

    class _Wrapper:
        def __init__(self, model):
            self.model = model

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            pred = self.model.predict(X.values)[0.5]
            prob = 1.0 / (1.0 + np.exp(-pred))
            return np.column_stack([1 - prob, prob])

    return _Wrapper(nq)


_TRAINERS = {
    "lightgbm": _train_lightgbm,
    "cross_asset_transformer": _train_cross_asset_transformer,
    "neural_quantile": _train_neural_quantile,
}


def main(
    cfg: dict | None = None,
    data: Tuple[pd.DataFrame, Iterable[int]] | None = None,
) -> Dict[str, float]:
    """Train base learners and an ensemble model.

    Parameters
    ----------
    cfg:
        Optional configuration dictionary. If ``None`` the global config is loaded.
    data:
        Tuple of feature DataFrame and label iterable. If ``None`` the function
        expects data loading to be handled externally.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    import contextlib
    import types

    try:  # pragma: no cover - optional dependency
        from analytics import mlflow_client as mlflow
        if not hasattr(mlflow, "log_dict"):  # provide passthrough if missing
            import mlflow as _mlf  # type: ignore

            mlflow.log_dict = lambda data, name: _mlf.log_dict(data, name)
    except Exception:  # noqa: BLE001 - provide a lightweight stub for tests
        mlflow = types.SimpleNamespace(
            start_run=lambda *a, **k: contextlib.nullcontext(),
            log_dict=lambda *a, **k: None,
            log_metric=lambda *a, **k: None,
            log_metrics=lambda *a, **k: None,
            end_run=lambda *a, **k: None,
        )

    try:  # pragma: no cover - optional dependency
        from models.ensemble import EnsembleModel
    except Exception:  # noqa: BLE001 - simple fallback used in tests
        class EnsembleModel:  # type: ignore[no-redef]
            def __init__(self, models, meta_model=None):
                self.models = models
                self.meta_model = meta_model

            def predict(self, X, y=None):  # noqa: D401 - mimic interface
                preds = {n: m.predict_proba(X)[:, 1] for n, m in self.models.items()}
                if self.meta_model is not None:
                    meta_X = np.column_stack([preds[n] for n in self.models])
                    preds["ensemble"] = self.meta_model.predict_proba(meta_X)[:, 1]
                return preds

    if cfg is None:
        from utils import load_config

        cfg = load_config()
    ens_cfg = cfg.get("ensemble") or {}
    if not ens_cfg.get("enabled"):
        raise ValueError("Ensemble training is disabled in the configuration")
    if data is None:
        raise ValueError("Training data must be provided")
    X, y = data
    X_train, X_val, y_train, y_val = train_test_split(
        X, np.asarray(list(y)), test_size=0.5, random_state=42
    )
    models: Dict[str, Any] = {}
    val_preds: Dict[str, np.ndarray] = {}
    metrics: Dict[str, float] = {}
    coefficients: Dict[str, float] = {}
    div_weights: Dict[str, float] | None = None

    mlflow.start_run("ensemble_training", cfg)
    try:
        for name, mcfg in ens_cfg.get("base_models", {}).items():
            mtype = mcfg.get("type")
            params = mcfg.get("params")
            feats = mcfg.get("features")
            trainer = _TRAINERS.get(mtype)
            if trainer is None:
                raise ValueError(f"Unknown model type: {mtype}")
            X_tr = X_train[feats] if feats else X_train
            X_vl = X_val[feats] if feats else X_val
            model = trainer(X_tr, y_train, params)
            wrapped = _WrappedModel(model, feats)
            prob = wrapped.predict_proba(X_vl)[:, 1]
            models[name] = wrapped
            val_preds[name] = prob
            f1 = f1_score(y_val, (prob > 0.5).astype(int))
            metrics[name] = f1
            mlflow.log_metric(f"f1_{name}", f1)
            ic = information_coefficient(prob, y_val)
            coefficients[name] = ic
            mlflow.log_metric(f"ic_{name}", ic)

        if len(val_preds) > 1:
            corr = error_correlation_matrix(val_preds, y_val)
            if hasattr(mlflow, "log_dict"):
                mlflow.log_dict(corr.to_dict(), "error_correlation.json")
            corr_abs = corr.abs()
            np.fill_diagonal(corr_abs.values, np.nan)
            with np.errstate(all="ignore"):
                mean_corr_val = np.nanmean(corr_abs.values)
            mlflow.log_metric(
                "mean_error_correlation",
                float(np.nan_to_num(mean_corr_val, nan=0.0)),
            )
            if ens_cfg.get("diversity_weighting"):
                with np.errstate(all="ignore"):
                    mean_corr = np.nanmean(corr_abs.values, axis=1)
                diversity = np.clip(1.0 - mean_corr, 0.0, None)
                if not np.allclose(diversity.sum(), 0.0):
                    div_w = diversity / diversity.sum()
                    div_weights = {name: float(w) for name, w in zip(corr.index, div_w)}
                    if hasattr(mlflow, "log_dict"):
                        mlflow.log_dict(div_weights, "diversity_weights.json")

        coef_vals = np.array([coefficients[n] for n in models])
        if np.any(~np.isfinite(coef_vals)):
            weights = {n: 1.0 / len(models) for n in models}
        else:
            coef_vals = np.abs(coef_vals)
            if np.allclose(coef_vals.sum(), 0.0):
                weights = {n: 1.0 / len(models) for n in models}
            else:
                exp_vals = np.exp(coef_vals)
                total = float(exp_vals.sum())
                weights = {n: float(v / total) for n, v in zip(models, exp_vals)}
        if div_weights is not None:
            weights = {n: weights[n] * div_weights[n] for n in weights}
            total = sum(weights.values())
            if total > 0:
                weights = {n: w / total for n, w in weights.items()}
            else:
                weights = {n: 1.0 / len(models) for n in models}
        if hasattr(mlflow, "log_dict"):
            mlflow.log_dict(coefficients, "information_coefficients.json")
            mlflow.log_dict(weights, "ensemble_weights.json")

        meta_model = None
        if ens_cfg.get("meta_learner"):
            meta_X = np.column_stack([val_preds[n] for n in models])
            meta_model = LogisticRegression().fit(meta_X, y_val)

        ensemble = EnsembleModel(models, weights=weights, meta_model=meta_model)
        all_preds = ensemble.predict(X_val)
        ens_prob = all_preds["ensemble"]
        ens_f1 = f1_score(y_val, (ens_prob > 0.5).astype(int))
        metrics["ensemble"] = ens_f1
        mlflow.log_metric("f1_ensemble", ens_f1)
        best_base = max(metrics[n] for n in models)
        mlflow.log_metric("ensemble_improvement", ens_f1 - best_base)
        if cfg.get("use_price_distribution"):
from mt5.train_price_distribution import train_price_distribution

            _, dist_metrics = train_price_distribution(
                X_train.values,
                y_train,
                X_val.values,
                y_val,
                n_components=int(cfg.get("n_components", 3)),
                epochs=int(cfg.get("dist_epochs", 100)),
            )
            mlflow.log_metrics({
                "dist_coverage": dist_metrics["coverage"],
                "dist_baseline_coverage": dist_metrics["baseline_coverage"],
                "dist_expected_shortfall": dist_metrics["expected_shortfall"],
            })
    finally:
        mlflow.end_run()

    return metrics


if __name__ == "__main__":  # pragma: no cover
    metrics = main()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
