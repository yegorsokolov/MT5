# Migration Workflow

This guide outlines how to migrate the trading bot and its state to a new VPS.

## Export state

1. **Stop the service** on the current VPS to ensure no files change during the export.
2. Run the export script to bundle checkpoints, data and configuration. Additional
   paths may be included or excluded with `--include`/`--exclude` options:
   ```bash
   scripts/export_state.sh --include /extra/path --exclude '*.tmp'
   ```
   Alternatively, use the dashboard's **Export state** button to generate and download
   the archive.
3. Transfer the resulting archive to the new VPS using `scp` or any preferred method.

## Import state on the new VPS

1. Copy the exported archive to the new server.
2. Stop the service on the new VPS.
3. Restore checkpoints and configuration:
   ```bash
   scripts/import_state.sh /path/to/project_state_<timestamp>.tar.gz
   ```
   Use `--include` for any extra paths that were exported. The dashboard also provides an
   **Import state** button to upload the archive directly.
4. Start the service on the new VPS. The latest checkpoint will be loaded automatically.

Importing overwrites existing checkpoints and configuration, replacing any current training progress.

## Wipe and redeploy a VPS

When you need to completely erase a VPS and rebuild it from scratch, reuse the
cleanup helper that already ships with the repository before running the
standard deployment script again.

1. **Stop the MT5 services** (replace the service name if you customised it):

   ```bash
   sudo systemctl stop mt5bot mt5bot-update mt5bot-update.timer
   ```

2. **Run the erase helper**. It unregisters the systemd units, removes the
   installation directory under `/opt/mt5`, deletes the Wine prefixes and wipes
   cached data. Adjust `SERVICE_NAME`/`INSTALL_ROOT` if your installation uses
   different paths:

   ```bash
   chmod +x scripts/mt5_cleanup.sh
   SERVICE_NAME=mt5bot INSTALL_ROOT=/opt/mt5 ./scripts/mt5_cleanup.sh
   ```

3. **Reinstall the stack** using the deployment script to restore a clean
   baseline:

   ```bash
   chmod +x scripts/deploy_mt5.sh
   ./scripts/deploy_mt5.sh
   ```

4. **Follow the on-screen prompts** to log into the MetaTrader 5 terminal, then
   press **Enter** in the SSH session so the script can finish configuring the
   environment.

These commands leave the VPS in the same state as a fresh provision, ready to
import checkpoints or start new training runs.
