from pathlib import Path
import sys
import types
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies so plugin package can import
sys.modules.setdefault("log_utils", types.SimpleNamespace(setup_logging=lambda *a, **k: None))
sys.modules.setdefault("metrics", types.SimpleNamespace(ERROR_COUNT=None, TRADE_COUNT=None))
sys.modules.setdefault(
    "crypto_utils",
    types.SimpleNamespace(
        _load_key=lambda *a, **k: b"", encrypt=lambda x: x, decrypt=lambda x: x
    ),
)
sys.modules.setdefault(
    "analytics.metrics_store", types.SimpleNamespace(record_metric=lambda *a, **k: None)
)

class _Caps:
    def capability_tier(self):
        return "lite"
    cpus = 4
    memory_gb = 8.0
    has_gpu = False

class _Monitor:
    capabilities = _Caps()
    capability_tier = "lite"

    def start(self):
        pass

    def subscribe(self):  # pragma: no cover - not used
        import asyncio

        return asyncio.Queue()

sys.modules.setdefault("utils", types.SimpleNamespace(resource_monitor=types.SimpleNamespace(monitor=_Monitor())))
sys.modules.setdefault("utils.resource_monitor", types.SimpleNamespace(monitor=_Monitor()))

from plugins import multilang_sentiment


def test_multilang_sentiment_translation(monkeypatch):
    df = pd.DataFrame(
        {
            "headline": [
                "Stocks rally on good news",
                "Las acciones caen por malas noticias",
            ],
            "lang": ["en", "es"],
        }
    )

    class FakeEnPipe:
        def __call__(self, texts):
            out = []
            for t in texts:
                if "rally" in t or "good" in t:
                    out.append({"label": "positive", "score": 0.9})
                else:
                    out.append({"label": "negative", "score": 0.8})
            return out

    def fake_translate(text: str, src: str) -> str:
        mapping = {"Las acciones caen por malas noticias": "stocks fall on bad news"}
        return mapping.get(text, text)

    monkeypatch.setattr(multilang_sentiment, "_get_xlm_pipeline", lambda mode=None: None)
    monkeypatch.setattr(multilang_sentiment, "_get_en_pipeline", lambda: FakeEnPipe())
    monkeypatch.setattr(multilang_sentiment, "_translate", fake_translate)

    out = multilang_sentiment.score_headlines(df)
    assert out.loc[0, "sentiment"] > 0
    assert out.loc[1, "sentiment"] < 0


def test_multilang_sentiment_xlm(monkeypatch):
    df = pd.DataFrame(
        {
            "headline": [
                "bonne nouvelle pour les marchés",
                "schlechte wirtschaftsdaten",
            ],
            "lang": ["fr", "de"],
        }
    )

    class FakePipe:
        def __call__(self, texts):
            return [
                {"label": "POSITIVE", "score": 0.7},
                {"label": "NEGATIVE", "score": 0.9},
            ]

    monkeypatch.setattr(multilang_sentiment, "_get_xlm_pipeline", lambda mode=None: FakePipe())

    out = multilang_sentiment.score_headlines(df)
    assert out.loc[0, "sentiment"] > 0
    assert out.loc[1, "sentiment"] < 0
