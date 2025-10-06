# Fix mt5linux Wine launcher

[Start Task](#launch-steps){ .md-button .md-button--primary }

## Summary
- Ensure the Wine-side mt5linux server is launched with the required positional Python argument so the CLI no longer exits with `the following arguments are required: python`.
- Prefer a translated Windows path derived from `MT5LINUX_BOOTSTRAP_PYTHON` when provided, otherwise fall back to the detected Windows interpreter.
- Keep the bridge log free from `ucrtbase.dll.crealf` crashes by continuing to wrap Wine invocations with native UCRT overrides.

## Acceptance criteria
- Running `./install_programmatic_bridge.sh` starts the mt5linux RPyC server without argument errors.
- The helper logs show the Windows Python path appended to the launch command and the server stays running.
- The script still honours optional overrides such as `MT5LINUX_BOOTSTRAP_PYTHON`.

## Launch steps {#launch-steps}
1. Pull the latest changes.
2. From the repository root run `./install_programmatic_bridge.sh` (set `PY_WINE_PREFIX`/`MT5_WINE_PREFIX` if they differ from defaults).
3. Confirm `/home/<user>/.wine-*/drive_c/mt5linux-server/mt5linux.log` contains the "Server started" message.

## Additional notes
- If the server log reports missing native CRT DLLs, rerun `winetricks -q vcrun2022` inside the affected prefix before retrying the task.
- For troubleshooting, manually invoke `wine "C:\\Program Files\\Python311\\python.exe" -m mt5linux --help` to confirm the CLI parses arguments as expected.
