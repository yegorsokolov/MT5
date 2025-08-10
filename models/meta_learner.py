import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Iterable

from . import model_store


class MetaLearner:
    """Fine-tune a pre-trained model for a new symbol.

    The learner freezes all layers except the last child module and
    trains remaining parameters on a small dataset.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @staticmethod
    def _load_latest_state_dict(symbol: str) -> dict | None:
        """Return the latest state_dict for a symbol from the model store."""
        try:
            versions = model_store.list_versions()
        except Exception:  # pragma: no cover - store may not exist
            return None
        for meta in reversed(versions):
            cfg = meta.get("training_config", {})
            syms = cfg.get("symbols") or [cfg.get("symbol")]
            if syms and symbol in syms:
                state, _ = model_store.load_model(meta["version_id"])
                if isinstance(state, dict):
                    return state
        return None

    @classmethod
    def from_symbol(
        cls, symbol: str, builder: Callable[[], torch.nn.Module]
    ) -> "MetaLearner":
        """Construct a MetaLearner initialised from a donor symbol."""
        model = builder()
        state_dict = cls._load_latest_state_dict(symbol)
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        return cls(model)

    def freeze_shared_layers(self) -> None:
        children = list(self.model.children())
        for child in children[:-1]:
            for p in child.parameters():
                p.requires_grad = False

    def fine_tune(
        self,
        dataset: TensorDataset,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str | None = None,
    ) -> list[float]:
        """Fine-tune the model and return accuracy per epoch."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.freeze_shared_layers()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        loss_fn = torch.nn.BCELoss()
        X_all, y_all = dataset.tensors
        history: list[float] = []
        for _ in range(epochs):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimiser.zero_grad()
                out = self.model(X)
                loss = loss_fn(out, y)
                loss.backward()
                optimiser.step()
            with torch.no_grad():
                preds = (self.model(X_all.to(device)) > 0.5).float()
                acc = (preds.cpu() == y_all).float().mean().item()
            history.append(acc)
        return history
