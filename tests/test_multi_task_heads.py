import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from data.feature_scaler import FeatureScaler
import train


def _train_single_task_baseline(X: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
    label_cols = [c for c in y.columns if c.startswith("direction_")]
    abs_cols = [c for c in y.columns if c.startswith("abs_return_")]
    vol_cols = [c for c in y.columns if c.startswith("volatility_")]

    steps: list[tuple[str, object]] = []
    steps.append(("scaler", FeatureScaler()))
    clf = MultiOutputClassifier(LGBMClassifier(n_estimators=10, random_state=0))
    steps.append(("clf", clf))
    pipe = Pipeline(steps)
    if label_cols:
        pipe.fit(X, y[label_cols])
    else:
        pipe.fit(X, y)

    reports: dict[str, float] = {}
    if label_cols:
        preds = pipe.predict(X)
        f1_scores: list[float] = []
        for i, col in enumerate(label_cols):
            rep = classification_report(y[col], preds[:, i], output_dict=True)
            f1_scores.append(rep["weighted avg"]["f1-score"])
        reports["aggregate_f1"] = float(np.mean(f1_scores))
    else:
        reports["aggregate_f1"] = 0.0

    trunk = pipe[:-1]
    X_reg = trunk.transform(X) if len(pipe.steps) > 1 else X
    if isinstance(X_reg, pd.DataFrame):
        X_reg = X_reg.to_numpy(dtype=float)
    else:
        X_reg = np.asarray(X_reg, dtype=float)

    rmse_scores: list[float] = []
    abs_rmse: list[float] = []
    vol_rmse: list[float] = []
    if abs_cols:
        reg_abs = MultiOutputRegressor(LinearRegression())
        reg_abs.fit(X_reg, y[abs_cols])
        pred_abs = reg_abs.predict(X_reg)
        if np.ndim(pred_abs) == 1:
            pred_abs = pred_abs.reshape(-1, 1)
        for i, col in enumerate(abs_cols):
            rmse = float(np.sqrt(np.mean((pred_abs[:, i] - y[col].to_numpy()) ** 2)))
            rmse_scores.append(rmse)
            abs_rmse.append(rmse)
    if vol_cols:
        reg_vol = MultiOutputRegressor(LinearRegression())
        reg_vol.fit(X_reg, y[vol_cols])
        pred_vol = reg_vol.predict(X_reg)
        if np.ndim(pred_vol) == 1:
            pred_vol = pred_vol.reshape(-1, 1)
        for i, col in enumerate(vol_cols):
            rmse = float(np.sqrt(np.mean((pred_vol[:, i] - y[col].to_numpy()) ** 2)))
            rmse_scores.append(rmse)
            vol_rmse.append(rmse)

    reports["aggregate_abs_return_rmse"] = float(np.mean(abs_rmse)) if abs_rmse else 0.0
    reports["aggregate_volatility_rmse"] = float(np.mean(vol_rmse)) if vol_rmse else 0.0
    if rmse_scores:
        reports["aggregate_rmse"] = float(np.mean(rmse_scores))
    else:
        reports["aggregate_rmse"] = 0.0
    return reports


def test_multi_task_training_outperforms_single_task_baseline():
    rng = np.random.default_rng(42)
    n = 240
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["feat_0", "feat_1", "feat_2"])
    latent = X["feat_0"] * X["feat_1"] + 0.4 * X["feat_2"]
    abs_return = np.abs(latent) + rng.normal(scale=0.05, size=n)
    volatility = 0.6 * np.abs(latent) + rng.normal(scale=0.04, size=n)
    logits = latent + 0.25 * abs_return + rng.normal(scale=0.1, size=n)
    direction = (logits > 0).astype(int)

    y = pd.DataFrame(
        {
            "direction_1": direction,
            "abs_return_1": abs_return,
            "volatility_1": volatility,
        }
    )

    baseline_metrics = _train_single_task_baseline(X, y)

    multi_cfg = {
        "use_scaler": False,
        "head_epochs": 400,
        "head_learning_rate": 0.05,
        "head_hidden_dim": 32,
        "head_classification_weight": 1.0,
        "head_abs_weight": 1.0,
        "head_volatility_weight": 1.0,
    }
    _, multi_metrics = train.train_multi_output_model(X, y, multi_cfg)

    assert multi_metrics["aggregate_f1"] > baseline_metrics["aggregate_f1"]
    assert (
        multi_metrics["aggregate_abs_return_rmse"]
        < baseline_metrics["aggregate_abs_return_rmse"]
    )
    assert (
        multi_metrics["aggregate_volatility_rmse"]
        < baseline_metrics["aggregate_volatility_rmse"]
    )
