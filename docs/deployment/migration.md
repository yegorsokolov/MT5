# Migration Workflow

This guide outlines how to migrate the trading bot and its state to a new VPS.

## Export state

1. **Stop the service** on the current VPS to ensure no files change during the export.
2. Run the export script to bundle checkpoints, data and configuration:
   ```bash
   scripts/export_state.sh
   ```
3. Transfer the resulting archive to the new VPS using `scp` or any preferred method.

## Import state on the new VPS

1. Copy the exported archive to the new server.
2. Restore checkpoints and configuration:
   ```bash
   scripts/import_state.sh /path/to/project_state_<timestamp>.tar.gz
   ```
3. Start the service on the new VPS. The latest checkpoint will be loaded automatically.

## Copy through the dashboard

The web dashboard includes a **Copy project** button in the sidebar. Clicking it generates the archive and offers it for download, which can then be moved to another environment and restored using the import script above.
