from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "self_review", ROOT / "strategy" / "self_review.py"
)
self_review = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(self_review)
self_review_strategy = self_review.self_review_strategy


def dummy_template(goal: str, context: str, config):
    return {"evaluate": context, "plan": goal, "act": "do"}


def test_self_review_cycle(tmp_path: Path):
    strategy = {"goal": "g", "context": "c", "evaluate": "e", "plan": "p", "act": "a"}
    log_dir = tmp_path / "reviews"
    final = self_review_strategy(strategy, dummy_template, log_dir, metrics_path=tmp_path / "metrics.parquet")
    assert isinstance(final, dict)
    assert (log_dir / "draft_0.json").exists()
    assert (log_dir / "draft_1.json").exists()
    assert (log_dir / "draft_2.json").exists()
    assert "Review 2" in final["context"]
