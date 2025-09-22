# Run history exports

This directory stores structured metadata and supporting artefacts for each
training or evaluation run. Entries are written by
`reports.run_history.RunHistoryRecorder` and are designed to be human readable so
that Codex reviewers can inspect the run context directly from the repository.

Each run produces:

- `<run_id>/run.json` — JSON summary of metrics, configuration and attached
  artefacts.
- Optional artefact copies such as log excerpts or reports in the same
  `<run_id>` folder.
- `index.json` — Manifest of recorded runs, updated after every run.
- `latest.json` — Pointer to the most recent run for convenience.

The directory is committed to Git so that run histories produced during local or
CI executions can be shared and reviewed.
