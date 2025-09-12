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
from analysis.information_coefficient import information_coefficient


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


def build_moe(cfg: dict | None = None) -> tuple[GatingNetwork, np.ndarray]:
    """Construct base experts and a gating network.

    Parameters
    ----------
    cfg:
        Optional configuration providing ``expert_weights`` (list of floats)
        and ``sharpness`` for the gating network's softmax. Defaults are equal
        weights and a sharpness of ``5.0``.
    """

    cfg = cfg or {}
    sharpness = float(cfg.get("sharpness", 5.0))
    expert_weights = np.asarray(cfg.get("expert_weights", [1.0, 1.0, 1.0]), dtype=float)

    experts = [
        ExpertSpec(TrendExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
        ExpertSpec(MeanReversionExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
        ExpertSpec(MacroExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
    ]

    return GatingNetwork(experts, sharpness=sharpness), expert_weights


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

    gating, expert_weights = build_moe(cfg)
    weights = gating.weights(market_regime, caps) * expert_weights
    total = weights.sum()
    if total > 0:
        weights = weights / total
    preds = np.array([spec.model.predict(history) for spec in gating.experts])
    return float(np.dot(weights, preds))

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
            mlflow.log_metric(
                "mean_error_correlation", float(np.nanmean(corr_abs.values))
            )
            if ens_cfg.get("diversity_weighting"):
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
            from train_price_distribution import train_price_distribution

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
