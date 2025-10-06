# Fix mt5linux Wine launcher

[Start Task](#launch-steps){ .md-button .md-button--primary }

## Summary
- Launch the mt5linux RPyC server under Wine with the required positional Python argument so the CLI no longer exits with `the following arguments are required: python`.
- Prefer the Windows path derived from `MT5LINUX_BOOTSTRAP_PYTHON` when provided, otherwise fall back to the detected interpreter path.
- Keep Wine invocations wrapped with native UCRT overrides to avoid `ucrtbase.dll.crealf` crashes.

## Acceptance criteria
- `./install_programmatic_bridge.sh` starts the mt5linux RPyC server without argument errors.
- The bridge log shows the Windows Python path appended to the launch command and the server remains running.
- Optional overrides like `MT5LINUX_BOOTSTRAP_PYTHON` continue to work.

## Launch steps {#launch-steps}
1. Pull the latest changes.
2. From the repository root run `./install_programmatic_bridge.sh` (set `PY_WINE_PREFIX`/`MT5_WINE_PREFIX` if they differ from defaults).
3. Confirm the server log under `~/.wine-*/drive_c/mt5linux-server/mt5linux.log` reports that the RPyC listener is active.

## Additional notes
- If the server log reports missing native CRT DLLs, rerun `winetricks -q vcrun2022` inside the affected prefix before retrying the task.
- For troubleshooting, manually invoke `wine "C:\\Program Files\\Python311\\python.exe" -m mt5linux --help` to validate the CLI wiring.
