import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

torch = pytest.importorskip("torch")

from models.cross_modal_classifier import CrossModalClassifier
from models import model_store
from generate_signals import load_models


def _make_modal_frame(n_samples: int = 240, window: int = 5, news_dim: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    price = rng.standard_normal((n_samples + window,))
    data: dict[str, np.ndarray] = {}
    for step in range(window):
        data[f"price_window_{step}"] = price[step : step + n_samples]
    news = rng.standard_normal((n_samples, news_dim))
    for dim in range(news_dim):
        data[f"news_emb_{dim}"] = news[:, dim]
    price_signal = sum(data[f"price_window_{step}"] for step in range(window)) / window
    news_signal = news.mean(axis=1)
    labels = ((price_signal > 0) & (news_signal > 0)).astype(float)
    frame = pd.DataFrame(data)
    frame["return"] = rng.normal(scale=0.01, size=n_samples)
    frame["market_regime"] = np.zeros(n_samples, dtype=int)
    frame["tb_label"] = labels
    return frame


def test_cross_modal_classifier_beats_flat_baseline():
    frame = _make_modal_frame()
    split = int(len(frame) * 0.75)
    train_df = frame.iloc[:split]
    test_df = frame.iloc[split:]
    features = [c for c in frame.columns if c != "tb_label"]
    clf = CrossModalClassifier(d_model=32, nhead=2, num_layers=2, epochs=40, lr=0.01)
    clf.fit(train_df[features], train_df["tb_label"].to_numpy())
    preds = clf.predict(test_df[features])
    cross_f1 = f1_score(test_df["tb_label"], preds)

    flat_features = [c for c in features if c.startswith("price_window_") or c.startswith("news_emb_")]
    baseline = LogisticRegression(max_iter=500)
    baseline.fit(train_df[flat_features], train_df["tb_label"])
    base_preds = baseline.predict(test_df[flat_features])
    base_f1 = f1_score(test_df["tb_label"], base_preds)

    assert cross_f1 > base_f1


def test_cross_modal_model_roundtrip(tmp_path, monkeypatch):
    frame = _make_modal_frame(n_samples=200)
    features = [c for c in frame.columns if c != "tb_label"]
    clf = CrossModalClassifier(d_model=16, nhead=2, num_layers=2, epochs=30, lr=0.01)
    clf.fit(frame[features], frame["tb_label"].to_numpy())

    monkeypatch.setattr(model_store, "STORE_DIR", tmp_path)
    version = model_store.save_model(
        clf,
        {"model_type": "cross_modal"},
        {"val_loss": 0.0, "test_accuracy": 1.0},
        features=features,
    )

    models, loaded_features, _ = load_models([], [version], return_meta=True)
    assert models, "Expected at least one model to be loaded"
    loaded = models[0]
    np.testing.assert_array_equal(sorted(features), sorted(loaded_features))

    test_probs = loaded.predict_proba(frame[features])
    direct_probs = clf.predict_proba(frame[features])
    np.testing.assert_allclose(test_probs, direct_probs, atol=1e-6)
