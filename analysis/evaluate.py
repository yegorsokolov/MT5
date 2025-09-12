import logging
from typing import Iterable, Tuple, Dict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def bootstrap_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    n_bootstrap: int = 1000,
    seed: int | None = 42,
    average: str = "binary",
    zero_division: int = 0,
) -> Dict[str, Tuple[float, float] | float]:
    """Compute metrics with 95% confidence intervals via bootstrapping.

    Parameters
    ----------
    y_true, y_pred:
        Iterable of true and predicted labels.
    n_bootstrap:
        Number of bootstrap resamples to draw.
    seed:
        Random seed for reproducibility.
    average:
        Averaging method for :mod:`sklearn` metrics.
    zero_division:
        Passed through to the metric functions.

    Returns
    -------
    dict
        Dictionary containing the point estimates and (lower, upper) 95%%
        confidence intervals for precision, recall and F1.
    """

    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    n = len(y_true_arr)
    if n == 0:
        raise ValueError("No samples provided for evaluation")

    rng = np.random.default_rng(seed)
    stats = np.empty((n_bootstrap, 3), dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        stats[i, 0] = precision_score(
            yt, yp, average=average, zero_division=zero_division
        )
        stats[i, 1] = recall_score(yt, yp, average=average, zero_division=zero_division)
        stats[i, 2] = f1_score(yt, yp, average=average, zero_division=zero_division)

    means = stats.mean(axis=0)
    lower = np.percentile(stats, 2.5, axis=0)
    upper = np.percentile(stats, 97.5, axis=0)

    logger.info("Precision %.3f (95%% CI %.3f-%.3f)", means[0], lower[0], upper[0])
    logger.info("Recall %.3f (95%% CI %.3f-%.3f)", means[1], lower[1], upper[1])
    logger.info("F1 %.3f (95%% CI %.3f-%.3f)", means[2], lower[2], upper[2])

    return {
        "precision": float(means[0]),
        "recall": float(means[1]),
        "f1": float(means[2]),
        "precision_ci": (float(lower[0]), float(upper[0])),
        "recall_ci": (float(lower[1]), float(upper[1])),
        "f1_ci": (float(lower[2]), float(upper[2])),
    }


def risk_adjusted_metrics(
    y_true: Iterable[int], y_pred: Iterable[int]
) -> Dict[str, float]:
    """Compute simple risk-adjusted metrics from predictions.

    The helper derives a pseudo return series from ``y_true`` and ``y_pred``
    by assuming a unit position is taken when the prediction is ``1``.
    Correct predictions yield ``+1`` while incorrect positives result in
    ``-1``.  No trade corresponds to a zero return.  From this return
    sequence the Sharpe and Calmar ratios are calculated.

    Parameters
    ----------
    y_true, y_pred:
        Iterable of true labels and binary predictions.

    Returns
    -------
    dict
        Dictionary with ``sharpe`` and ``calmar`` ratios.
    """

    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=int)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")

    returns = np.where(yp == 1, np.where(yt == 1, 1.0, -1.0), 0.0)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = float(mean_ret / (std_ret + 1e-9))

    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    max_dd = np.max(peak - cumulative) if len(cumulative) else 0.0
    calmar = float(mean_ret / (max_dd + 1e-9))

    logger.info("Sharpe %.3f | Calmar %.3f", sharpe, calmar)

    return {"sharpe": sharpe, "calmar": calmar}
