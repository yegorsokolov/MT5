import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from analytics.issue_client import IssueClient
from analytics import decision_logger


def test_issue_client_local(tmp_path):
    path = tmp_path / "issues.json"
    client = IssueClient(repo_path=path)
    iid = client.post_event("drift", {"detail": "x"})
    assert iid is not None
    issues = client.list_open()
    assert any(i["id"] == iid for i in issues)
    assert client.update_status(iid, "closed")
    assert client.list_open() == []


def test_decision_logger_with_issue(monkeypatch):
    captured = {}

    def fake_log_decision(df):
        captured["df"] = df

    monkeypatch.setattr(decision_logger, "log_decision", fake_log_decision)
    df = pd.DataFrame({"event": ["prediction"]})
    decision_logger.log(df, issue_ids=["ISSUE-1"]) 
    assert "issues" in captured["df"].columns
    assert captured["df"]["issues"].iloc[0] == ["ISSUE-1"]


def test_replay_strategies_issue(monkeypatch, tmp_path):
    from analysis import replay
    import log_utils
    import types, sys

    decisions = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01")],
        "return": [0.1],
        "volatility": [0.2],
        "trend_strength": [0.3],
        "regime": [1],
        "market_basket": [0],
        "issues": [["ISS-1"]],
    })
    monkeypatch.setattr(log_utils, "read_decisions", lambda: decisions, raising=False)

    class DummyAlgo:
        def __call__(self, feats):
            return 1

    class DummyRouter:
        def __init__(self):
            self.algorithms = {"a": DummyAlgo()}
            self.scoreboard_path = tmp_path / "scoreboard.parquet"

    router_mod = types.ModuleType("strategy.router")
    router_mod.StrategyRouter = lambda: DummyRouter()
    sys.modules["strategy.router"] = router_mod
    sys.modules.setdefault("strategy", types.ModuleType("strategy")).router = router_mod
    monkeypatch.setattr(replay, "STRATEGY_REPLAY_DIR", tmp_path)
    replay.replay_strategies(["a"])
    out_file = tmp_path / "a.parquet"
    if out_file.exists():
        out_df = pd.read_parquet(out_file)
    else:
        out_df = pd.read_csv(tmp_path / "a.csv")
    assert "issues" in out_df.columns
    val = out_df["issues"].iloc[0]
    if isinstance(val, str):
        assert "ISS-1" in val
    else:
        assert val == ["ISS-1"]
