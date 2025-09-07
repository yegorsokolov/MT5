"""Train multiple models and build an ensemble."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple, Any

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from analytics import mlflow_client as mlflow
from models.ensemble import EnsembleModel
from models.quantile_regression import NeuralQuantile
from lightgbm import LGBMClassifier

from utils import load_config

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
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = net(X_t).view(-1, 1)
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
                logits = self.net(x).view(-1)
                prob = torch.sigmoid(logits).numpy()
            return np.column_stack([1 - prob, prob])

    return _Wrapper(net)


def _train_neural_quantile(
    X: pd.DataFrame, y: np.ndarray, params: Mapping[str, Any] | None
) -> Any:
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

    if cfg is None:
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

        meta_model = None
        if ens_cfg.get("meta_learner"):
            meta_X = np.column_stack([val_preds[n] for n in models])
            meta_model = LogisticRegression().fit(meta_X, y_val)

        ensemble = EnsembleModel(models, meta_model=meta_model)
        all_preds = ensemble.predict(X_val, y_val)
        ens_prob = all_preds["ensemble"]
        ens_f1 = f1_score(y_val, (ens_prob > 0.5).astype(int))
        metrics["ensemble"] = ens_f1
        mlflow.log_metric("f1_ensemble", ens_f1)
        best_base = max(metrics[n] for n in models)
        mlflow.log_metric("ensemble_improvement", ens_f1 - best_base)
    finally:
        mlflow.end_run()

    return metrics


if __name__ == "__main__":  # pragma: no cover
    metrics = main()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
