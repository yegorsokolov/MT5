from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from tests.plugins import log_archive as log_archive_plugin


pytest_plugins = ("pytester",)


@pytest.mark.usefixtures("tmp_path")
def test_log_archive_plugin_captures_logs(pytester, monkeypatch):
    repo_root = log_archive_plugin._find_repo_root(Path(__file__).resolve())
    commit = log_archive_plugin._current_commit(repo_root)
    logs_dir = repo_root / "logs"
    logs_dir_preexisting = logs_dir.exists()
    logs_dir.mkdir(exist_ok=True)

    unique_suffix = uuid.uuid4().hex
    test_log = logs_dir / f"log-archive-test-{unique_suffix}.log"
    test_log.write_text("log archive plugin test", encoding="utf-8")

    extra_dir = repo_root / f"runtime-log-archive-{unique_suffix}"
    extra_dir.mkdir(exist_ok=True)
    extra_file = extra_dir / "runtime.log"
    extra_file.write_text("runtime temp data", encoding="utf-8")

    monkeypatch.setenv(log_archive_plugin.ENV_EXTRA_DIRS, str(extra_dir))

    commit_dir = repo_root / "logs" / "test_runs" / commit
    marker_file = commit_dir / log_archive_plugin.LATEST_MARKER
    commit_dir_preexisting = commit_dir.exists()
    previous_marker = None
    if marker_file.exists():
        previous_marker = marker_file.read_text(encoding="utf-8").strip()

    existing_runs: set[str] = set()
    if commit_dir.exists():
        existing_runs = {
            path.name for path in commit_dir.iterdir() if path.is_dir()
        }

    pytester.makepyfile(
        """
        def test_dummy():
            assert True
        """
    )

    new_run_dir: Path | None = None
    try:
        result = pytester.runpytest_inprocess("-p", "tests.plugins.log_archive", "-q")
        result.assert_outcomes(passed=1)

        assert commit_dir.exists()
        new_runs = [
            path for path in commit_dir.iterdir() if path.is_dir() and path.name not in existing_runs
        ]
        assert new_runs, "expected the log archive plugin to create a new run directory"

        new_runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        new_run_dir = new_runs[0]

        archived_log = new_run_dir / "logs" / test_log.name
        assert archived_log.exists()
        assert archived_log.read_text(encoding="utf-8") == "log archive plugin test"

        archived_extra = new_run_dir / "extras" / extra_dir.name / extra_file.name
        assert archived_extra.exists()
        assert archived_extra.read_text(encoding="utf-8") == "runtime temp data"

        assert (
            marker_file.read_text(encoding="utf-8").strip() == new_run_dir.name
        )
    finally:
        if new_run_dir and new_run_dir.exists():
            shutil.rmtree(new_run_dir)

        if previous_marker is not None:
            marker_file.write_text(previous_marker, encoding="utf-8")
        elif marker_file.exists():
            marker_file.unlink()

        if commit_dir.exists():
            remaining_runs = [path for path in commit_dir.iterdir() if path.is_dir()]
            if not remaining_runs and not commit_dir_preexisting:
                if marker_file.exists():
                    marker_file.unlink()
                commit_dir.rmdir()

        if extra_dir.exists():
            shutil.rmtree(extra_dir)

        if test_log.exists():
            test_log.unlink()

        if not logs_dir_preexisting and logs_dir.exists():
            try:
                logs_dir.rmdir()
            except OSError:
                # Ignore if other log files or directories were present during the test run.
                pass


def test_log_archive_plugin_clears_previous_failures(pytester):
    repo_root = log_archive_plugin._find_repo_root(Path(__file__).resolve())
    commit = log_archive_plugin._current_commit(repo_root)
    logs_dir = repo_root / "logs"
    logs_dir_preexisting = logs_dir.exists()
    logs_dir.mkdir(exist_ok=True)

    unique_suffix = uuid.uuid4().hex
    guard_log = logs_dir / f"log-archive-cleanup-{unique_suffix}.log"
    guard_log.write_text("guard", encoding="utf-8")

    summary_path = logs_dir / "test_failures.json"
    existing_summary = None
    if summary_path.exists():
        existing_summary = summary_path.read_text(encoding="utf-8")

    test_runs_dir = logs_dir / "test_runs"
    test_runs_preexisting = test_runs_dir.exists()
    commit_dir = logs_dir / "test_runs" / commit
    commit_dir_preexisting = commit_dir.exists()
    preexisting_runs = set()
    if commit_dir.exists():
        preexisting_runs = {path.name for path in commit_dir.iterdir() if path.is_dir()}

    created_run_dirs: list[Path] = []

    try:
        fail_test = pytester.makepyfile(
            test_failure="""
            def test_failure():
                assert False
            """
        )
        result = pytester.runpytest_inprocess("-p", "tests.plugins.log_archive", "-q")
        result.assert_outcomes(failed=1)

        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["failures"], "expected failures to be recorded"

        fail_test.unlink()

        commit_dir.mkdir(parents=True, exist_ok=True)
        new_runs = [
            path
            for path in commit_dir.iterdir()
            if path.is_dir() and path.name not in preexisting_runs
        ]
        assert new_runs, "expected a run directory to be created for the failure"
        for run_dir in new_runs:
            created_run_dirs.append(run_dir)
            preexisting_runs.add(run_dir.name)
            failure_record = run_dir / "failures.json"
            assert failure_record.exists()

        success_test = pytester.makepyfile(
            test_success="""
            def test_success():
                assert True
            """
        )
        result = pytester.runpytest_inprocess("-p", "tests.plugins.log_archive", "-q")
        result.assert_outcomes(passed=1)

        assert not summary_path.exists()

        success_test.unlink()

        post_runs = [
            path
            for path in commit_dir.iterdir()
            if path.is_dir() and path.name not in preexisting_runs
        ]
        for run_dir in post_runs:
            created_run_dirs.append(run_dir)
            preexisting_runs.add(run_dir.name)

        for run_dir in created_run_dirs:
            failure_record = run_dir / "failures.json"
            assert not failure_record.exists()
    finally:
        if existing_summary is not None:
            summary_path.write_text(existing_summary, encoding="utf-8")
        elif summary_path.exists():
            summary_path.unlink()

        for run_dir in created_run_dirs:
            if run_dir.exists():
                shutil.rmtree(run_dir)

        if not commit_dir_preexisting and commit_dir.exists():
            try:
                commit_dir.rmdir()
            except OSError:
                pass

        if not test_runs_preexisting and test_runs_dir.exists():
            try:
                test_runs_dir.rmdir()
            except OSError:
                pass

        if guard_log.exists():
            guard_log.unlink()

        if not logs_dir_preexisting and logs_dir.exists():
            try:
                logs_dir.rmdir()
            except OSError:
                pass
