"""Sklearn-style wrapper around :class:`CrossModalTransformer`."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency during docs builds
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch is optional in some environments
    torch = None  # type: ignore[assignment]
    TensorDataset = DataLoader = None  # type: ignore[assignment]

from sklearn.base import BaseEstimator, ClassifierMixin
from mt5.train_utils import prepare_modal_arrays

if torch is not None:  # pragma: no cover - torch may be unavailable during docs
    from .cross_modal_transformer import CrossModalTransformer
else:  # pragma: no cover - handled by guard clauses at runtime
    CrossModalTransformer = None  # type: ignore[assignment]


class CrossModalClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier powered by :class:`CrossModalTransformer`."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        time_encoding: bool = False,
        average_attn_weights: bool = True,
        device: str | None = None,
    ) -> None:
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.time_encoding = bool(time_encoding)
        self.average_attn_weights = bool(average_attn_weights)
        self.device = device

        self.price_columns_: list[str] | None = None
        self.news_columns_: list[str] | None = None
        self.price_dim_: int | None = None
        self.news_dim_: int | None = None
        self.constant_prob_: float = 0.5
        self.state_dict_: dict[str, torch.Tensor] | None = None  # type: ignore[assignment]
        self.feature_names_in_: np.ndarray | None = None

        self._model: CrossModalTransformer | None = None  # type: ignore[assignment]
        self._device: torch.device | None = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # sklearn API helpers
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "time_encoding": self.time_encoding,
            "average_attn_weights": self.average_attn_weights,
            "device": self.device,
        }

    def set_params(self, **params: Any) -> "CrossModalClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # training / inference utilities
    # ------------------------------------------------------------------
    def _ensure_torch(self) -> None:
        if torch is None or CrossModalTransformer is None or TensorDataset is None:
            raise RuntimeError(
                "PyTorch is required for CrossModalClassifier but is not available"
            )

    def _normalise_input(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            if self.feature_names_in_ is None:
                raise ValueError("Feature names are unknown; fit the model before use")
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        return df

    def _init_model(self, device: torch.device) -> CrossModalTransformer:
        if self.price_dim_ is None or self.news_dim_ is None:
            raise RuntimeError("Model dimensions are unknown; fit must be called first")
        model = CrossModalTransformer(
            price_dim=int(self.price_dim_),
            news_dim=int(self.news_dim_),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=1,
            time_encoding=self.time_encoding,
            average_attn_weights=self.average_attn_weights,
        )
        return model.to(device)

    def _ensure_model(self) -> CrossModalTransformer | None:
        if self.state_dict_ is None:
            return None
        if self._model is not None:
            return self._model
        self._ensure_torch()
        if self.device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(self.device)
        model = self._init_model(dev)
        model.load_state_dict({k: v.to(dev) for k, v in self.state_dict_.items()})
        model.eval()
        self._model = model
        self._device = dev
        return model

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> "CrossModalClassifier":
        self._ensure_torch()
        self.state_dict_ = {k: v.detach().cpu() for k, v in state_dict.items()}
        self._model = None
        self._device = None
        return self

    def _train_loader(
        self,
        price: np.ndarray,
        news: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> DataLoader:
        tensors: list[torch.Tensor] = [
            torch.tensor(price, dtype=torch.float32),
            torch.tensor(news, dtype=torch.float32),
            torch.tensor(labels.astype(np.float32), dtype=torch.float32),
        ]
        if sample_weight is not None:
            tensors.append(torch.tensor(sample_weight.astype(np.float32), dtype=torch.float32))
        dataset = TensorDataset(*tensors)  # type: ignore[arg-type]
        batch = max(1, min(self.batch_size, len(dataset)))
        return DataLoader(dataset, batch_size=batch, shuffle=True)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[float] | np.ndarray,
        sample_weight: Iterable[float] | np.ndarray | None = None,
    ) -> "CrossModalClassifier":
        self._ensure_torch()
        df = self._normalise_input(X)
        self.feature_names_in_ = np.asarray(df.columns)
        arrays = prepare_modal_arrays(df, np.asarray(y, dtype=np.float32))
        if arrays is None:
            self.state_dict_ = None
            self.constant_prob_ = float(np.mean(y)) if len(df) else 0.5
            return self
        price, news, labels, mask, price_cols, news_cols = arrays
        if labels is None:
            labels = np.asarray(y, dtype=np.float32)[mask]
        if len(price) == 0:
            self.state_dict_ = None
            self.constant_prob_ = float(np.mean(y)) if len(df) else 0.5
            return self

        self.price_columns_ = list(price_cols)
        self.news_columns_ = list(news_cols)
        self.price_dim_ = price.shape[-1]
        self.news_dim_ = news.shape[-1]
        self.constant_prob_ = float(np.clip(labels.mean(), 1e-6, 1 - 1e-6))

        weight_clean = None
        if sample_weight is not None:
            weight_arr = np.asarray(sample_weight, dtype=np.float32)
            if weight_arr.shape[0] != df.shape[0]:
                raise ValueError("sample_weight must match number of rows in X")
            weight_clean = weight_arr[mask]
        loader = self._train_loader(price, news, labels, sample_weight=weight_clean)

        if self.device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(self.device)
        model = self._init_model(dev)
        opt = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.BCELoss(reduction="none")

        for _ in range(max(1, self.epochs)):
            model.train()
            for batch in loader:
                if weight_clean is not None:
                    price_batch, news_batch, target, weights = batch  # type: ignore[misc]
                    weights = weights.to(dev)
                else:
                    price_batch, news_batch, target = batch  # type: ignore[misc]
                    weights = None
                price_batch = price_batch.to(dev)
                news_batch = news_batch.to(dev)
                target = target.to(dev)
                opt.zero_grad()
                preds = model(price_batch, news_batch)
                loss_raw = loss_fn(preds, target)
                if weights is not None:
                    norm = weights.sum()
                    if float(norm) <= 0:
                        loss = loss_raw.mean()
                    else:
                        loss = (loss_raw * weights).sum() / norm
                else:
                    loss = loss_raw.mean()
                loss.backward()
                opt.step()

        model.eval()
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self.state_dict_ = state
        self._model = None
        self._device = None
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        df = self._normalise_input(X)
        n = len(df)
        if n == 0:
            return np.zeros((0, 2), dtype=float)
        arrays = prepare_modal_arrays(df)
        if arrays is None or self.state_dict_ is None:
            probs = np.full(n, self.constant_prob_, dtype=float)
            return np.column_stack([1 - probs, probs])
        price, news, _, mask, _, _ = arrays
        model = self._ensure_model()
        if model is None:
            probs = np.full(n, self.constant_prob_, dtype=float)
            return np.column_stack([1 - probs, probs])
        device = self._device or torch.device("cpu")
        with torch.no_grad():
            price_tensor = torch.tensor(price, dtype=torch.float32, device=device)
            news_tensor = torch.tensor(news, dtype=torch.float32, device=device)
            preds = model(price_tensor, news_tensor).detach().cpu().numpy().reshape(-1)
        probs = np.full(n, self.constant_prob_, dtype=float)
        probs[mask] = preds
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - probs, probs])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        if probs.ndim == 2:
            probs = probs[:, 1]
        return (probs > 0.5).astype(int)

    # ------------------------------------------------------------------
    # pickling helpers
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_model", None)
        state.pop("_device", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._model = None
        self._device = None


__all__ = ["CrossModalClassifier"]
