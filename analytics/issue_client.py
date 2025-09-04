from __future__ import annotations

"""Client for recording structured issue events.

This lightweight client stores issue reports either by posting to a remote
HTTP API or by appending to a local JSON file.  The local file approach acts
as a simple git-based tracker allowing multiple VMs to share context without a
separate service.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

import pandas as pd
import requests


DEFAULT_PATH = Path("analytics/issues.json")


@dataclass
class IssueClient:
    """Post and query issues from a central tracker.

    Parameters
    ----------
    base_url:
        Optional HTTP endpoint.  If provided issues are created via REST
        requests.  When ``None`` a local JSON file is used instead, suitable
        for git-based synchronisation across hosts.
    repo_path:
        Location of the JSON file when ``base_url`` is ``None``.
    """

    base_url: str | None = None
    repo_path: Path = DEFAULT_PATH
    session: requests.Session = field(default_factory=requests.Session)

    # ------------------------------------------------------------------
    def _file_issues(self) -> List[Dict[str, Any]]:
        if self.repo_path.exists():
            try:
                return json.loads(self.repo_path.read_text())
            except Exception:
                return []
        return []

    # ------------------------------------------------------------------
    def post_event(
        self, event: str, details: Dict[str, Any], *, severity: str = "info"
    ) -> Optional[str]:
        """Create an issue and return its identifier.

        Parameters
        ----------
        event:
            Short event type, e.g. ``drift_alert``.
        details:
            Additional structured information describing the issue.
        severity:
            Severity level for the tracker.  Defaults to ``"info"``.
        """

        payload = {
            "event": event,
            "details": details,
            "severity": severity,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        if self.base_url:
            try:  # pragma: no cover - network
                resp = self.session.post(
                    self.base_url.rstrip("/") + "/issues",
                    json=payload,
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                return str(data.get("id")) if isinstance(data, dict) else None
            except Exception:
                return None
        issues = self._file_issues()
        issue_id = str(len(issues) + 1)
        payload.update({"id": issue_id, "status": "open"})
        issues.append(payload)
        self.repo_path.parent.mkdir(parents=True, exist_ok=True)
        self.repo_path.write_text(json.dumps(issues))
        return issue_id

    # ------------------------------------------------------------------
    def list_open(self) -> List[Dict[str, Any]]:
        """Return currently open issues."""
        if self.base_url:
            try:  # pragma: no cover - network
                resp = self.session.get(
                    self.base_url.rstrip("/") + "/issues",
                    params={"state": "open"},
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "issues" in data:
                    return data["issues"]  # type: ignore[return-value]
                if isinstance(data, list):
                    return data  # type: ignore[return-value]
                return []
            except Exception:
                return []
        return [i for i in self._file_issues() if i.get("status") != "closed"]

    # ------------------------------------------------------------------
    def update_status(self, issue_id: str, status: str) -> bool:
        """Update the status of an existing issue.

        Only supported in file mode.  Returns ``True`` if the update succeeded.
        """

        if self.base_url:
            try:  # pragma: no cover - network
                resp = self.session.patch(
                    self.base_url.rstrip("/") + f"/issues/{issue_id}",
                    json={"status": status},
                    timeout=5,
                )
                resp.raise_for_status()
                return True
            except Exception:
                return False
        issues = self._file_issues()
        updated = False
        for issue in issues:
            if str(issue.get("id")) == str(issue_id):
                issue["status"] = status
                updated = True
                break
        if updated:
            self.repo_path.write_text(json.dumps(issues))
        return updated


def load_default() -> IssueClient:
    """Return an ``IssueClient`` configured from environment variables."""

    url = os.getenv("ISSUE_TRACKER_URL")
    path = Path(os.getenv("ISSUE_TRACKER_PATH", DEFAULT_PATH))
    return IssueClient(base_url=url, repo_path=path)
