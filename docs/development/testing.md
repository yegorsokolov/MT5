# Testing workflow and log archival

The CI pipeline now captures the logs generated during each pytest invocation
and stores them in a structured location under the repository's `logs/`
directory. This section explains how the plugin works, how to run the helper
script that packages logs, and how Codex reviewers can retrieve the resulting
artefacts from GitHub Actions.

## Pytest log archive plugin

A lightweight pytest plugin (`tests/plugins/log_archive.py`) runs at the end of
every test session. It copies the contents of `logs/` (excluding previous
archives) as well as pytest's runtime temporary directory into
`logs/test_runs/<commit>/<timestamp>-<suffix>/` where `<commit>` is the current
`git rev-parse --short HEAD` hash. Each run is timestamped in UTC and the most
recent run identifier is stored in `logs/test_runs/<commit>/LATEST_RUN`.

The plugin supports a few customization hooks:

- Set the environment variable `PYTEST_DISABLE_LOG_ARCHIVE=1` or pass the
  `--no-log-archive` option to pytest to disable the behaviour.
- Define the ini option `log_archive_extra_dirs` or provide a
  `PYTEST_LOG_ARCHIVE_EXTRA_DIRS` environment variable (colon-separated list) to
  include additional directories in the archive, e.g. external services or
  custom temporary folders.

The plugin is loaded automatically from `tests/conftest.py`, so no extra
configuration is required when running pytest locally.

## Packaging logs with `tools/archive_test_logs.py`

The `tools/archive_test_logs.py` script compresses one of the collected runs or
an entire commit's worth of log snapshots. Example usage:

```bash
# Compress the most recent run for the current commit to a tar.gz
python tools/archive_test_logs.py

# Bundle every archived run for the current commit into zip format
python tools/archive_test_logs.py --scope commit --format zip

# Package a specific run ID into a custom location
python tools/archive_test_logs.py --run-id 20240101T120000Z-abcd1234 \
    --output /tmp/mt5-test-logs.tar.gz
```

When executed inside GitHub Actions the script writes the archive path to
`$GITHUB_OUTPUT`, allowing later steps to upload the artefact. The archive is
created in `logs/test_runs/<commit>-<identifier>.tar.gz` by default and remains
inside the repository for manual inspection.

## CI workflow updates

`.github/workflows/tests.yaml` now runs the archive script after the testing
steps. The archive is uploaded as a workflow artefact named
`test-logs-<commit>` so Codex reviewers can download the logs directly from the
workflow run page. Because the archival step is wrapped in `if: always()`, logs
are still preserved even when earlier test steps fail.

To retrieve logs locally, download the artefact from GitHub Actions and extract
it into the repository. Each run is stored in its own subdirectory under
`logs/test_runs/<commit>/`, mirroring the directories captured during the test
session.
