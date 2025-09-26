# Deployment Guide

The live trading stack depends on three separate applications:

* **MetaTrader 5 Terminal** – streams market data and executes orders.
* **Git/GitHub** – stores configuration, datasets and trained models so they can be
  synchronised across machines.
* **Python runtime** – runs the training loop, signal publisher and
  automation scripts contained in this repository.

Use the checklist below to provision a clean Windows machine or VPS.

## 1. Install prerequisites

1. **MetaTrader 5 Terminal**
   - Download the installer from your broker or the official MetaTrader
     website and run it with the default options.
   - After installation, launch the terminal once to let it create the
     `MQL5` data directory.
   - Log in with your trading account so the Python trading backend can execute
     live or demo trades.
2. **Git and GitHub tooling**
   - Install Git from [git-scm.com](https://git-scm.com/download/win). Accept the
     defaults and ensure “Git from the command line” is selected so the bot can
     run scheduled commits.
   - (Optional) Install [GitHub Desktop](https://desktop.github.com/) or the
     [GitHub CLI](https://cli.github.com/) if you prefer a guided sign-in flow.
     Sign in with your GitHub account and grant it access to the repository so
     the automation scripts can push logs and checkpoints.
   - Configure your identity in Git by running `git config --global user.name
     "<Your Name>"` and `git config --global user.email "<you@example.com>"`.
3. **Python environment**
   - Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/)
     and tick **Add Python to PATH**.
   - Open “Command Prompt” and verify the installation with `python --version`
     and `pip --version`.

## 2. Clone and prepare the repository

1. Clone the project either through GitHub Desktop (“File → Clone
   repository”) or by running `git clone <repo-url>` inside Git Bash.
2. Open a command prompt in the cloned folder and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The repository now ships a single consolidated `requirements.txt` that bundles
   the previously separate dependency groups. Trim the file or install extras via
   `pyproject.toml` if you prefer a leaner environment. Install any extras you
   require, for example `pip install .[heavy]`.
3. Generate the Protobuf bindings the signal publisher and trading backend
   consume:

   ```bash
   protoc --python_out=. proto/signals.proto
   ```

## 3. Connect the bot to MetaTrader 5

1. Launch the MetaTrader 5 terminal, log into the account you intend to trade
   and keep the terminal running so the Python bridge can reuse the session.
2. Ensure Python can discover the terminal by setting the `MT5_TERMINAL_PATH`
   environment variable (point it at the installation directory or directly at
   `terminal64.exe`) or by placing the terminal inside the repository `mt5/`
   folder.
3. Run `python scripts/setup_terminal.py --install-heartbeat` (add
   `--path "<terminal-or-install-dir>"` if the helper cannot auto-detect the
   executable). Provide `--login`, `--password` and `--server` to perform a
   headless login or omit them to attach to the running terminal. The script
   reports the connected account, balance and broker when successful.
4. If the script fails, review the printed MetaTrader5 error and run the
   installed `ConnectionHeartbeat` script inside MetaTrader 5 for detailed
   diagnostics before retrying.
5. Finally, open **Tools → Options → Expert Advisors** inside MetaTrader 5 and
   enable **Allow automated trading**, then ensure the **Algo Trading** toggle on
   the toolbar is green so the terminal accepts orders from Python.

## 4. Run the automation scripts

1. Publish predictions with `python -m mt5.generate_signals`.
2. Start real-time training with `python -m mt5.realtime_train`.
3. To keep GitHub in sync, schedule `python scripts/hourly_artifact_push.py`
   and ensure your Git credentials are cached so pushes succeed without manual
   prompts.
