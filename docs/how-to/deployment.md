# Deployment Guide

The live Expert Advisor depends on three separate applications:

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
   - Log in with your trading account so the Expert Advisor can trade live or
     on a demo environment.
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
   pip install -r requirements-core.txt
   ```

   Install any extras you require, for example `pip install .[heavy]`.
3. Generate the Protobuf bindings the Expert Advisor consumes:

   ```bash
   protoc --python_out=. proto/signals.proto
   ```

## 3. Connect the bot to MetaTrader 5

1. Copy the Expert Advisor files into the terminal by running
   `python scripts/setup_terminal.py "<path-to-terminal>"` or manually copying
   `AdaptiveEA.mq5` and `RealtimeEA.mq5` into `MQL5/Experts`.
2. Restart MetaTrader 5, open the MetaEditor and press **F7** to compile the
   new Expert Advisors.
3. In MetaTrader 5 enable **Tools → Options → Expert Advisors → Allow automated
   trading** and toggle the **Algo Trading** button on the toolbar.
4. Drag `AdaptiveEA` (or `RealtimeEA`) from the **Navigator** onto a chart and
   verify the `ZmqAddress` input matches the URL used by the Python signal
   publisher (`tcp://localhost:5555` by default).
5. Check the **Experts** tab for a “Connected to signal socket” message – this
   confirms the bot is receiving signals from Python.

## 4. Run the automation scripts

1. Publish predictions with `python -m mt5.generate_signals`.
2. Start real-time training with `python -m mt5.realtime_train`.
3. To keep GitHub in sync, schedule `python scripts/hourly_artifact_push.py`
   and ensure your Git credentials are cached so pushes succeed without manual
   prompts.
