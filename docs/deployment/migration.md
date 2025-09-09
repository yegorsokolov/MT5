# Migration Workflow

This guide outlines how to migrate the trading bot and its state to a new VPS.

## Export state

1. **Stop the service** on the current VPS to ensure no files change during the export.
2. Run the export script to bundle checkpoints, data and configuration:
   ```bash
   scripts/export_state.sh
   ```
   Alternatively, use the dashboard's **Export state** button to generate and download the archive.
3. Transfer the resulting archive to the new VPS using `scp` or any preferred method.

## Import state on the new VPS

1. Copy the exported archive to the new server.
2. Stop the service on the new VPS.
3. Restore checkpoints and configuration:
   ```bash
   scripts/import_state.sh /path/to/project_state_<timestamp>.tar.gz
   ```
   The dashboard also provides an **Import state** button to upload the archive directly.
4. Start the service on the new VPS. The latest checkpoint will be loaded automatically.

Importing overwrites existing checkpoints and configuration, replacing any current training progress.
