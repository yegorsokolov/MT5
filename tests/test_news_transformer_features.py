from pathlib import Path
import importlib.util
import pandas as pd
import types
import sys

spec = importlib.util.spec_from_file_location(
    "news", Path(__file__).resolve().parents[1] / "features" / "news.py"
)
sys.modules["yaml"] = types.SimpleNamespace(
    safe_load=lambda *a, **k: {}, safe_dump=lambda *a, **k: "", dump=lambda *a, **k: ""
)
sys.modules.setdefault("data.events", types.SimpleNamespace(get_events=lambda **_: []))
news = importlib.util.module_from_spec(spec)
spec.loader.exec_module(news)


def test_embeddings_shape(monkeypatch):
    # Use a distilled sentiment model
    monkeypatch.setattr(
        news, "MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
    )
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2024-01-01", periods=2, freq="H"),
        "news_summary": ["stocks rally on earnings", "market tumbles on data"],
    })
    enriched = news.add_news_sentiment_features(df.copy())
    emb_cols = [c for c in enriched.columns if c.startswith("news_emb_")]
    assert emb_cols, "embedding columns missing"
    # Dimension should match model hidden size (distilbert -> 768)
    assert len(emb_cols) == 768
    assert enriched[emb_cols].shape == (2, len(emb_cols))
    assert "news_sentiment" in enriched.columns
