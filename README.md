# Adaptive MT5 Trading Bot

This folder contains a modular Expert Advisor (EA) and supporting Python scripts for training and
backtesting machine learning driven trading strategies on MetaTrader 5.

## Overview

The solution is split into two components:

1. **Python toolkit** — handles data preprocessing, feature engineering, machine
   learning and backtesting using the provided CSV history files.
2. **MQL5 Expert Advisor** — a lightweight EA that loads model signals and
   executes trades with a dynamic trailing stop while respecting FTMO style risk
    limits.  Risk controls enforce maximum daily loss and overall drawdown in the
    EA itself and are configurable via `config.yaml`. Position sizing can
    automatically adjust based on recent volatility or the realised Sharpe ratio
    to better match current market conditions. Additional checks compute a
     99% value-at-risk and a simple stress-loss estimate so trading is paused
     when tail risk grows beyond the configured thresholds. The latest version
     also monitors conditional value-at-risk (expected shortfall) and applies a
     regime-switching volatility model so position sizes adapt when markets
     transition between calm and turbulent periods. A filtered VaR calculation
     now uses an exponentially weighted variance controlled by the `var_decay`
     parameter.
3. **Realtime trainer** — a Python script that fetches live ticks from MT5,
   incrementally retrains the model and commits updates to this GitHub repo.
4. **Auto optimiser** — uses **scikit-optimize** to search signal thresholds,
   window sizes and key RL parameters with cross‑validated walk‑forward
   backtests. Any improved settings are written back to `config.yaml` along with
   the reason in `logs/config_changes.csv` and logged to MLflow.
### Risk management

Key risk parameters in `config.yaml` include `max_daily_loss`, `max_drawdown`, `max_var`, `max_stress_loss`, `max_cvar` and `var_decay`, which controls the exponential weighting for the filtered VaR calculation.

The feature engineering step now includes additional indicators such as
lower/higher timeframe moving averages (e.g. the `ma_60` one‑hour average), a volatility measure and basic
 order-book statistics (spread and volume imbalance) and microstructure cues
 like depth imbalance, trade rate and quote revisions. Optional ATR, Donchian
 and Keltner channel calculations are provided via the `atr`, `donchian` and
`keltner` plugins. These plugins can be toggled with the `use_atr`,
`use_donchian` and `use_keltner` flags in `config.yaml`.
To enable the hidden Markov model based regime classifier set
`use_regime_classifier: true`.

```yaml
use_atr: true
use_donchian: true
use_keltner: true
use_regime_classifier: true
```
Spread and slippage protections are provided via the `spread` and `slippage`
plugins. Enable them with the `use_spread_check` and `use_slippage_check`
flags and configure thresholds through `max_spread` and `max_slippage`.
The dataset also merges
 high impact events from several economic calendars (ForexFactory, the built-in
 MetaTrader calendar via Tradays and the MQL5 feed) so the bot can avoid trading
 immediately around red news releases. These richer features help the model
capture more market behaviour than simple MAs and RSI alone.

The project can be adapted to any symbol by changing the configuration
parameters and retraining the model on the corresponding historical data.
`train.py` now supports training on multiple symbols at once.  By default both
`XAUUSD` and `GBPUSD` history files will be downloaded and combined.
The script `train_nn.py` now uses a lightweight Transformer network on sliding
windows of features for those wanting to explore deep learning models.
`train_meta.py` demonstrates a simple meta-learning approach where a global
model is fitted on all symbols and lightweight adapters are fine-tuned for each
instrument.  The per-symbol models are saved under the `models/` folder.
Another option `train_rl.py` trains a reinforcement learning agent that
optimises risk-adjusted profit.  The PPO environment now supports trading
multiple symbols at once using a vector of position sizes.  Per-symbol returns
and transaction costs are tracked while a portfolio variance penalty discourages
excess risk.  Key parameters such as `rl_max_position`, `rl_transaction_cost`,
`rl_risk_penalty` and `rl_var_window` can be adjusted in `config.yaml`.
For a full pipeline combining all of these approaches run `train_combined.py`.

## Installation

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place historical CSV files under `data/`, specify a mapping of symbols to their download URLs in `config.yaml` under `data_urls`, **or** define `api_history` entries to fetch ticks directly from your MetaTrader&nbsp;5 terminal. Existing CSV files can be converted to Parquet using `python scripts/migrate_to_parquet.py`.
   The MT5 history center provides free tick data once you have logged in to a broker through the terminal.
3. The realtime trainer stores ticks in a DuckDB database located at `data/realtime.duckdb`. The database is created automatically the first time you run the script and old rows beyond the `realtime_window` setting are pruned on each update.
4. Adjust settings in `config.yaml` if needed. The `symbols` list controls which instruments are used for training.
5. Train the model and run a backtest:

   ```bash
   python train.py
   # run the transformer-based neural network
   python train_nn.py
   # train symbol-specific adapters
   python train_meta.py
   python train_rl.py
   # end-to-end training of all components
   python train_combined.py
   python backtest.py
   ```

   If `data_urls` are provided, `train.py` will download the file(s) for the configured symbols via `gdown` before training.
   When `api_history` entries are present, the data will instead be pulled directly from the MetaTrader&nbsp;5 history center.

The resulting model file (`model.joblib`) can be loaded by the EA.

To run live training and keep the repository in sync:

```bash
python realtime_train.py
```
This script continuously pulls ticks from the terminal, retrains the model and
pushes the updated dataset and model back to the repository.

## Deployment Guide

Follow these steps to run the EA and the realtime trainer on a Windows PC or VPS:

1. **Provision a system** – If running remotely, create a Windows VPS with at
   least 2 GB of RAM. On a local Windows desktop you can skip this step.
2. **Install MetaTrader 5** –
   1. Open your browser on the VPS and download MetaTrader 5 from your broker or the official website.
   2. Run the installer and click **Next** until the installation completes.
3. **Install Git and Python** –
   1. Download Git from [git-scm.com](https://git-scm.com/) and install using the default options (click **Next** on each screen).
   2. Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
      During installation tick the **Add Python to PATH** checkbox and click **Install Now**.
4. **Clone this repository** –
   1. Launch **Git Bash** from the Start menu.
   2. Run `git clone <repo-url>` and press **Enter**.
5. **Install dependencies** –
   1. Open **Command Prompt** and `cd` into the cloned folder.
   2. Run `pip install -r requirements.txt`.
6. **Initial training** –
   1. Still inside the command prompt run `python train.py`.
      The script downloads the backtesting files `XAUUSD.csv` and `GBPUSD.csv`
      from Google Drive and trains a LightGBM model.
   2. To experiment with the transformer-based neural network instead run `python train_nn.py`.
      This trains a small transformer on sequences of the same features and saves `model_transformer.pt`.
   3. After either script finishes you will see the resulting model file under the project folder.
7. **Copy the EA** –
   1. Open MetaTrader 5 and click **File → Open Data Folder**.
   2. Run `python scripts/setup_terminal.py "<path-to-terminal>"` to automatically place `AdaptiveEA.mq5` and `RealtimeEA.mq5` inside `MQL5/Experts`.
   3. Restart MetaTrader 5 and compile the EA inside the MetaEditor by pressing **F7**.
8. **Attach the EA** –
   1. In MetaTrader 5 open the **Navigator** panel (Ctrl+N).
   2. Drag the EA onto a chart of either XAUUSD or GBPUSD and click **OK**.
9. **Publish signals** –
   1. In the command prompt run `python generate_signals.py`.
      This publishes prediction messages to `tcp://localhost:5555` which the EA subscribes to.
   2. Set the environment variable `SIGNAL_QUEUE_BIND` or `SIGNAL_QUEUE_URL` to change the port if needed.
10. **Run realtime training** –
   1. Back in the command prompt run `python realtime_train.py`.
   2. Leave this window open; the script will keep updating `model.joblib` as new ticks arrive.
11. **Optimise parameters** –
   1. Periodically run `python auto_optimize.py`.
      The optimiser performs a Bayesian search across thresholds, walk‑forward
      window sizes and reinforcement‑learning parameters. Results are
      cross‑validated over multiple market regimes and both the metrics and
      chosen hyperparameters are tracked with MLflow. Any improvements are
      written back to `config.yaml` and logged under `logs/config_changes.csv`.
12. **Upload logs** –
   1. Start `python scripts/hourly_log_push.py` in a separate window. This
      script commits and pushes the `logs/` folder every hour so log history is
      archived automatically. Use Windows Task Scheduler to launch it at logon
      for unattended operation.

13. **Keep it running** –
   1. Create scheduled tasks that start both `python realtime_train.py` and the
      hourly log uploader whenever the VPS boots or a user logs in. With these
      tasks enabled the bot and log push service run indefinitely.

With the EA running on your VPS and the training script collecting realtime data,
the bot will continually adapt to market conditions.

## MetaTrader 5 EA

The EA script `AdaptiveEA.mq5` demonstrates how to load predictions
produced by the Python model and place trades with a context aware trailing
stop.  `RealtimeEA.mq5` extends this idea by automatically running `git pull`
on initialisation so the latest model from GitHub is used.

Both experts expose a `ZmqAddress` input allowing you to change the
subscriber endpoint. By default this is `tcp://localhost:5555` and matches the
address used by `generate_signals.py`.

Both EAs subscribe to a ZeroMQ topic to receive probability signals from the
Python models. This decouples the EA from the file system so predictions arrive
in realtime without relying on `signals.csv` on disk. If the EA misses a
timestamp it simply uses the next received message.
The queue implementation now uses asynchronous sockets which further reduces
latency and eliminates file polling entirely.

`generate_signals.py` merges ML probabilities with a moving average
crossover and RSI filter so trades are only taken when multiple conditions
confirm the direction.  Additional optional filters check for Bollinger band
breakouts, volume spikes and even macro indicators when a `macro.csv` file is
present. Configuration values for these filters live in `config.yaml`.  The
pipeline now also considers news sentiment scores and cross-asset momentum
to further refine entries. Set `enable_news_trading` to `false` to automatically
block trades within a few minutes of scheduled high impact events pulled from
all three calendars.

An optional **advanced signal combination** stage can ensemble multiple models
stored under the `models/` folder. When `ensemble_models` is defined in
`config.yaml` the script loads each model and either averages the probabilities
or performs a simple Bayesian model averaging based on the `ensemble_method`
setting. When `blend_with_rl` is enabled the probabilities are stacked with
signals from the RL agent and a logistic regression is fit on the fly to
produce a combined forecast. This allows blending a baseline LightGBM model with
meta-learning, transformer outputs and reinforcement learning for more robust
entries.

## Performance Reports

`backtest.py` outputs statistics including win rate, Sharpe ratio and maximum
drawdown. These metrics can be used to iteratively optimise the strategy.

## Parallelized Training / HPC

For multiple symbols and large tick datasets, fitting models sequentially can
be slow. The script `train_parallel.py` uses Ray to distribute training and
backtests across available CPU cores or even a cluster. Enable this by setting
`use_ray: true` in `config.yaml` and optionally adjust `ray_num_cpus` to limit
resource usage. The Docker setup remains unchanged so experiments stay
reproducible.

## Plugin Architecture

Feature engineering functions, models and risk checks can now be extended via
plugins under the `plugins/` package. Register new components with the helper
decorators exposed in `plugins.__init__` and they will automatically be applied
when `dataset.make_features` or training scripts run.
Built-in examples include the `atr`, `donchian` and `keltner` plugins which add
ATR, Donchian and Keltner channel signals. A regime classification plugin can
also be enabled to label each row using a hidden Markov model.
Risk checks for spread limits and slippage detection are provided by the
`spread` and `slippage` modules.

## Strategy Templates

Example strategies adapted from the open source Freqtrade and Backtrader
frameworks are included under `strategies/`. They demonstrate how the feature
pipeline can feed different trading styles ranging from simple MA crossovers to
Donchian breakouts with ATR based stops.

## Detailed Logging

All scripts now log to `logs/app.log` with rotation to prevent the file from
growing indefinitely. The `log_utils` module also patches `print` so anything
printed to the console is captured in the log file. Key functions are wrapped
with a decorator to record start/end markers and any exceptions.

The helper script `scripts/upload_logs.py` can be run to automatically commit
and push the log directory to your repository for later analysis.

## Streamlined Deployment

The repository now includes a `Dockerfile` and GitHub Actions workflow which
mirror the manual Windows VPS steps. Building the container installs MetaTrader
5 under Wine, all Python dependencies and copies the EA so an identical
environment can be launched under WSL or Docker Desktop. Running
`docker-compose up` spins up the terminal with the latest code making rollbacks
trivial.  Containers start via `scripts/run_bot.sh` which performs an initial
training pass when no `model.joblib` is present, launches the terminal and then
enters the realtime training loop while uploading logs in the background.

The workflow `.github/workflows/train.yml` retrains both `train.py` and
`train_nn.py` whenever new data is pushed or on a daily schedule. Generated
models are committed back to the repository ensuring the EA always uses the
most recent versions.

## Kubernetes Deployment

For larger scale or multi‑VPS setups the bot can be run inside a Kubernetes
cluster. Example manifests are provided under `k8s/`. Deploy the persistent
volume claim and the deployment:

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
```

The deployment image also invokes `scripts/run_bot.sh` so pods automatically
train on first start and then switch to the realtime loop.

The container image should be pushed to a registry accessible by your cluster
(`ghcr.io/youruser/mt5-bot:latest` by default). The deployment mounts a
persistent volume at `/opt/mt5` so the MetaTrader terminal and models survive
pod restarts. Adjust the manifest to scale replicas or to integrate with your
cluster's ingress.

Liveness and readiness probes now call `scripts/healthcheck.py` so Kubernetes can
restart unhealthy pods. Configuration values such as the path to `config.yaml`
or the ZeroMQ address can be overridden via environment variables defined in the
deployment manifest.

## Remote Management API

A small FastAPI application defined in `remote_api.py` exposes REST endpoints for
starting and stopping multiple bots. Launch the server with:

```bash
uvicorn remote_api:app --host 0.0.0.0 --port 8000
```

Set `API_KEY` before launching and include the header `X-API-Key` in each
request. Use `GET /bots` to list running instances, `POST /bots/<id>/start`
and `POST /bots/<id>/stop` to control them. `GET /bots/<id>/status` checks a
single bot while `GET /logs` returns the last few lines from `logs/app.log`.
`GET /health` exposes overall service status and recent logs. `POST /config`
accepts JSON fields `key`, `value` and `reason` to update `config.yaml`.

**Note:** Keep this section updated whenever deployment scripts or automation
change to avoid configuration drift.

---

This is a simplified template intended for further extension and tuning.  It is
up to the user to verify performance using additional backtests and forward
testing within MetaTrader 5 before deploying to a live environment.

## Effect of Feature Scaling

The configuration file now includes a `use_scaler` flag to toggle a
`StandardScaler` step during model training. On the synthetic sample data
included with the tests the impact of scaling was minor:

| use_scaler | Sharpe | Max Drawdown | Total Return | Win Rate |
|-----------|-------|--------------|--------------|---------|
| True | 8.43 | -2.94% | 0.34 | 72% |
| False | 8.77 | -2.94% | 0.34 | 75% |

In practice the benefit of scaling will depend on the underlying market data
and features.
