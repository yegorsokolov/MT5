import numpy as np


def make_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Create focal loss objective for LightGBM."""

    def _focal_loss(y_pred: np.ndarray, data) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(data, np.ndarray):
            y_true, y_pred = y_pred, data
        elif hasattr(data, "get_label"):
            y_true = data.get_label()
        else:
            y_true = data
        p = 1.0 / (1.0 + np.exp(-y_pred))
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        p_t = p * y_true + (1 - p) * (1 - y_true)
        mod_factor = (1 - p_t) ** gamma
        grad = (p - y_true) * alpha_t * mod_factor
        hess = alpha_t * mod_factor * p * (1 - p)
        return grad, hess

    return _focal_loss


def make_focal_loss_metric(alpha: float = 0.25, gamma: float = 2.0):
    """Create focal loss evaluation metric for LightGBM."""

    def _focal_metric(y_pred: np.ndarray, data) -> tuple[str, float, bool]:
        if isinstance(data, np.ndarray):
            y_true, y_pred = y_pred, data
        elif hasattr(data, "get_label"):
            y_true = data.get_label()
        else:
            y_true = data
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = p * y_true + (1 - p) * (1 - y_true)
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = -alpha_t * (1 - p_t) ** gamma * np.log(np.clip(p_t, 1e-7, 1 - 1e-7))
        return "focal_loss", float(np.mean(loss)), False

    return _focal_metric
