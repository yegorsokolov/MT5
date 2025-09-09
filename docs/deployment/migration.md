# Migration Workflow

This guide outlines how to migrate the trading bot and its state to a new VPS.

## Export state

1. **Stop the service** on the current VPS to ensure no files change during the export.
2. Run the export script to bundle checkpoints, data and configuration into a
   single archive:
   ```bash
   scripts/export_state.sh
   ```
3. Transfer the resulting archive to the new VPS using `scp` or any preferred method.

## Import state on the new VPS

1. Copy the exported archive to the new server.
2. Restore checkpoints and configuration. The script removes any existing state
   so the imported archive fully replaces the current progress:
   ```bash
   scripts/import_state.sh /path/to/project_state_<timestamp>.tar.gz
   ```
3. Start the service on the new VPS. The latest checkpoint will be loaded automatically.

## Export and import through the dashboard

The web dashboard sidebar provides **Export state** and **Import state** buttons.
Use **Export state** to download an archive of the current model checkpoints and
configuration. On another instance, choose the file with **Import state** to
restore the data. Existing checkpoints and config will be overwritten and the
latest checkpoint is loaded automatically.
