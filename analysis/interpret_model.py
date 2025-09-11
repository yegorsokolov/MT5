"""Compute SHAP values for a trained model on recent data."""

from __future__ import annotations

from pathlib import Path
import logging
import joblib
import pandas as pd
import numpy as np

try:
    import shap
except Exception:  # pragma: no cover - handled in runtime
    shap = None
try:  # pragma: no cover - optional plotting
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)


def compute_shap_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return feature importance ranked by mean absolute SHAP value."""
    if shap is None:
        raise RuntimeError("shap is not installed")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if getattr(shap_values, "ndim", 2) == 3:  # newer shap returns (n, m, k)
        shap_values = shap_values[..., -1]
    importance = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def generate_shap_report(
    model,
    X: pd.DataFrame,
    report_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Create SHAP summary plot and ranked feature list.

    Parameters
    ----------
    model : fitted estimator or Pipeline
        Model used for SHAP value computation.  If a Pipeline is provided,
        the classifier and any preprocessing steps are handled automatically.
    X : pd.DataFrame
        Feature matrix used to compute SHAP values.
    report_dir : Path or str, optional
        Directory in which to store the plot and CSV report. Defaults to
        :data:`REPORT_DIR`.

    Returns
    -------
    dict[str, Path]
        Mapping containing paths to the generated ``plot`` and ``features``
        files.  Returns an empty dict if SHAP or matplotlib are unavailable
        or computation fails.
    """

    if shap is None or plt is None:
        logger.info("shap or matplotlib not installed, skipping SHAP report")
        return {}

    if hasattr(model, "named_steps"):
        pipe = model
        model = pipe.named_steps.get("clf", pipe)
        X_used = X
        if "scaler" in pipe.named_steps:
            X_used = pipe.named_steps["scaler"].transform(X_used)
    else:
        X_used = X

    try:
        fi = compute_shap_importance(model, X_used)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning("Failed to compute SHAP values: %s", exc)
        return {}

    if report_dir is None:
        report_dir = REPORT_DIR
    report_dir = Path(report_dir)
    report_dir.mkdir(exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_used)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plt.figure()
    shap.summary_plot(shap_values, X_used, show=False, plot_type="bar")
    plot_path = report_dir / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(plot_path)

    features_path = report_dir / "feature_importance.csv"
    fi.to_csv(features_path, index=False)
    return {"plot": plot_path, "features": features_path}


def main() -> None:
    """Load model and data then save a SHAP summary plot under ``reports/``."""
    if shap is None:
        print("shap not installed, skipping interpretation")
        return
    if plt is None:
        print("matplotlib not installed, skipping interpretation")
        return
    from data.history import load_history_parquet
    from data.features import make_features
    root = Path(__file__).resolve().parent.parent
    model_path = root / "model.joblib"
    data_path = root / "data" / "history.parquet"
    if not model_path.exists() or not data_path.exists():
        print("model or data missing")
        return
    pipe = joblib.load(model_path)
    df = load_history_parquet(data_path)
    df = make_features(df.tail(500))
    features = getattr(pipe, "feature_names_in_", df.columns.tolist())
    X = df[features]
    try:
        fi = compute_shap_importance(
            pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe, X
        )
    except Exception as exc:  # pragma: no cover - log warning and exit
        logger.warning("Failed to compute SHAP values: %s", exc)
        return
    explainer = shap.TreeExplainer(
        pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    )
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    out = REPORT_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out)
    fi.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print(f"Saved SHAP report to {out}")


if __name__ == "__main__":
    main()
