# Adaptive MT5 Trading Bot

This folder contains a modular Expert Advisor (EA) and supporting Python scripts for training and
backtesting machine learning driven trading strategies on MetaTrader 5.

## Overview

Detailed configuration options are documented in [docs/config.md](docs/config.md).

Commands used by the orchestrator to (re)start helper services are configurable via a `service_cmds` mapping in `config.yaml` or by setting the `SERVICE_COMMANDS` environment variable.

The solution is split into two components:

1. **Python toolkit** — handles data preprocessing, feature engineering, machine
   learning and backtesting using the provided CSV history files.
2. **Python trading backend** — uses the MetaTrader5 Python package directly
   for order execution. Risk controls such as maximum daily loss and
   drawdown limits remain configurable via ``config.yaml`` and are enforced
   before sending orders. Position sizing can automatically adjust based on
   recent volatility or the realised Sharpe ratio to better match current
   market conditions. Additional checks compute a 99% value‑at‑risk and a
   simple stress‑loss estimate so trading is paused when tail risk grows
   beyond the configured thresholds. The latest version also monitors
   conditional value‑at‑risk (expected shortfall) and applies a
   regime‑switching volatility model so position sizes adapt when markets
   transition between calm and turbulent periods. A filtered VaR calculation
   now uses an exponentially weighted variance controlled by the
   ``var_decay`` parameter.
   Historical ticks can be retrieved directly using
   ``brokers.mt5_direct.fetch_history`` for arbitrary symbols or
   ``brokers.mt5_direct.copy_ticks_from`` for fully Python-based backtesting.
3. **Realtime trainer** — a Python script that fetches live ticks from MT5,
   incrementally retrains the model and commits updates to this GitHub repo.
4. **Auto optimiser** — uses **scikit-optimize** to search signal thresholds,
   window sizes and key RL parameters with cross‑validated walk‑forward
   backtests. Any improved settings are written back to `config.yaml` along with
   the reason in `logs/config_changes.csv` and logged to MLflow.
5. **Feature evaluator** — iteratively tests optional features across rolling
   backtests and disables those that hurt performance. Run
   `python scripts/evaluate_features.py` after collecting enough data to update
   `config.yaml` automatically.
6. **Hyperparameter tuner** — pass `--tune` to the training scripts to launch
   an Optuna search over learning rate, model depth and RL discount factors. The
   best settings are logged to MLflow and persisted in `tuning/*.db`.

## Data sources

The feature engineering pipeline supports a range of optional external
datasets.  Fundamental loaders ingest company financial statements and
valuation ratios from local CSV files or lightweight APIs and merge them
with macroeconomic series such as GDP, CPI and interest rates.  A separate
alternative data loader integrates options‑implied volatility, blockchain
activity metrics and ESG scores when available.  These datasets reside
under `data/` or `dataset/` and are merged into the main feature table via
backward ``asof`` joins.

## Deployment and Environment Checks

The toolkit attempts to run even on minimal virtual machines. An environment
check verifies that all required Python packages from `requirements-core.txt`
are installed, checks available CPU cores and memory and adjusts the
configuration when the host has limited resources. If the VM does not meet the
minimum requirements the process aborts with a message describing the detected
hardware and the suggested specification (minimum 2 GB RAM and 1 CPU core;
recommended 8 GB RAM and 4 cores).

The check runs automatically whenever modules from `utils` are imported but can
also be invoked directly:

```bash
python -m utils.environment
```

This command raises an error if required packages are missing.

### CPU Feature Detection and Acceleration

At startup the resource monitor inspects `/proc/cpuinfo` for CPU flags such as
`avx2` and `fma`. When present, optional acceleration libraries are enabled
automatically. `numexpr` is used for fast array operations and pandas switches
to the PyArrow engine:

```python
pd.options.mode.dtype_backend = "pyarrow"
```

If the required flags or libraries are absent the project seamlessly falls back
to the standard code paths.

### Example deployment

Ubuntu:

```bash
git clone https://github.com/USERNAME/MT5.git
cd MT5
./scripts/setup_ubuntu.sh  # install system packages and core Python deps
dvc pull  # fetch raw/history data
python -m utils.environment  # verify deps and adjust config
python train.py
```

Set `WITH_CUDA=1` before running the script to install CUDA drivers when an
NVIDIA GPU is present.

For cloud deployments (EC2, Proxmox or other Ubuntu instances) supply
`deploy/cloud-init.yaml` as the user-data script. It installs apt
dependencies, runs `pip install -r requirements-core.txt` and enables the
`mt5bot.service` on first boot.

macOS:

```bash
git clone https://github.com/USERNAME/MT5.git
cd MT5
pip install -r requirements-core.txt
dvc pull  # fetch raw/history data
python -m utils.environment  # verify deps and adjust config
python train.py
```

Windows PowerShell:

```powershell
git clone https://github.com/USERNAME/MT5.git
Set-Location MT5
pip install -r requirements-core.txt
dvc pull  # fetch raw/history data
python -m utils.environment
python train.py
```

These commands download the repository, install required packages, verify the
environment and start a training run. Adjust the final command for backtesting
or signal generation as needed. Optional components can be installed via
extras, for example `pip install .[rl]` or `pip install .[heavy]`.

### Pre-commit hooks

Install the git hooks to automatically run formatting, linting and type checks:

```bash
pip install pre-commit && pre-commit install
```

After adding new data, track it with `dvc add` and upload it to the configured
remote using `dvc push`.

### Systemd service

To run the realtime trainer on boot, install the systemd unit:

```bash
sudo ./scripts/install_service.sh
```

This copies the service file to `/etc/systemd/system` and enables the service.
Common `systemctl` commands:

```bash
sudo systemctl start mt5bot     # start the service
sudo systemctl stop mt5bot      # stop the service
sudo systemctl restart mt5bot   # restart after updates
sudo systemctl status mt5bot    # check service status
journalctl -u mt5bot -f         # follow service logs
```

Application logs are also written to `logs/app.log` within the repository:

```bash
tail -f logs/app.log
```

Edit `deploy/mt5bot.service` to run `remote_api.py` instead of `realtime_train.py` if desired.

### Reproducibility

Set the `seed` value in `config.yaml` to reproduce training runs. The
training scripts will apply this seed to Python's `random` module, NumPy and
any framework-specific RNGs such as PyTorch:

```yaml
seed: 123
```

```bash
python train.py  # uses the seed from config.yaml
```

### Hyperparameter tuning

Each training script accepts a `--tune` flag which launches an Optuna search
over core parameters and logs the best trial to MLflow. Recommended ranges:

| Script | Parameters |
| ------ | ---------- |
| `train.py` | `learning_rate` 1e-4–0.2, `num_leaves` 16–255, `max_depth` 3–12 |
| `train_cli.py neural` | `learning_rate` 1e-5–1e-2, `d_model` 32–256, `num_layers` 1–4 |
| `train_rl.py` | `rl_learning_rate` 1e-5–1e-2, `rl_gamma` 0.90–0.999 |

### Masked time-series encoder pretraining

A compact GRU encoder can be pre-trained on historical windows to speed up
adaptation to new regimes. Run the helper script to populate the model store:

```bash
python pretrain_ts_encoder.py --config config.yaml
```

Training scripts will automatically load these weights when
`use_ts_pretrain` is enabled in `config.yaml`:

```yaml
use_ts_pretrain: true
ts_pretrain_epochs: 5
ts_pretrain_batch_size: 32
```

This initialisation typically reduces early training loss compared to random
weights.

### Contrastive encoder pretraining

A lightweight contrastive encoder can also be trained on unlabeled windows to
provide a useful initialization. Populate the model store with:

```bash
python pretrain_contrastive.py --config config.yaml
```

Training scripts load these weights automatically when
`use_contrastive_pretrain` is enabled in `config.yaml`:

```yaml
use_contrastive_pretrain: true
contrastive_epochs: 5
contrastive_batch_size: 32
```

This initialisation typically lowers early training loss versus random
initialisation.

Studies are stored in `tuning/*.db`. Rerun the script with `--tune` to resume an
interrupted optimisation or continue adding trials. Training checkpoints allow
each trial to pick up where it left off if terminated mid‑run.

### Risk management

Key risk parameters in `config.yaml` include `max_daily_loss`, `max_drawdown`, `max_var`, `max_stress_loss`, `max_cvar` and `var_decay`, which controls the exponential weighting for the filtered VaR calculation.  A portfolio level risk manager aggregates exposure and PnL across bots. Configure its limits with `max_portfolio_drawdown` and `max_var` (or set the environment variables `MAX_PORTFOLIO_DRAWDOWN` and `MAX_VAR`).

### Alerting

Alerts are raised for resource watchdog breaches, risk limit violations and data
drift. Configure a Slack webhook URL or SMTP credentials in `config.yaml` under
the `alerting` key:

```yaml
alerting:
  slack_webhook: https://hooks.slack.com/services/...
  smtp:
    host: smtp.example.com
    port: 587
    username: myuser
    password: mypass
    from: bot@example.com
    to: ops@example.com
```

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
use_deep_regime: true
```
The `deep_regime` plugin trains an LSTM autoencoder on rolling mid-price windows
and clusters the latent representations into discrete market states. To train and
use this detector:

1. Enable it in `config.yaml` with `use_deep_regime: true` (optional settings
   `deep_regime_window`, `deep_regime_dim` and `deep_regime_states` control the
   window length, embedding size and number of clusters).
2. Run `python dataset.py` or any training script. When no saved model is
   present the plugin fits the autoencoder and a k-means model, storing them
   under `models/deep_regime/`.
3. Subsequent feature generation merges the resulting labels into the dataset as
   the `regime_dl` column.
Additional technical factors can be generated via Microsoft **Qlib**.
First install `pyqlib` and then enable the plugin:

```bash
pip install pyqlib[all]
```

```yaml
use_qlib_features: true
use_gat_features: true
```
Time series characteristics from **tsfresh** can also be added. Install the
package and enable the plugin:

```bash
pip install tsfresh
```

```yaml
use_tsfresh: true
```
Macroeconomic series from the Federal Reserve's **FRED** database can be merged
as well. Install the data reader and enable the plugin:

```bash
pip install pandas_datareader
```

```yaml
use_fred_features: true
fred_series:
  - FEDFUNDS
  - GDP
```
Graph-based features can be extracted from cross-correlation networks when
`torch-geometric` is installed and the `graph_features` plugin is enabled. When
`use_gat_features` is set the plugin will train a small graph attention network
to embed each symbol at every timestamp. A temporal variant controlled by
`use_temporal_gat` processes adjacency matrices sequentially to generate
per-timestamp node embeddings.
The number of attention heads and dropout applied to the attention weights can
be configured via `gat_heads` and `gat_dropout`.
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
The `neural` subcommand in `train_cli.py` now uses a lightweight Transformer network on sliding
windows of features for those wanting to explore deep learning models.
`train_meta.py` demonstrates a simple meta-learning approach where a global
model is fitted on all symbols and lightweight adapters are fine-tuned for each
instrument.  The per-symbol models are saved under the `models/` folder.
Another option `train_rl.py` trains a reinforcement learning agent that
optimises risk-adjusted profit.  The PPO environment now supports trading
multiple symbols at once using a vector of position sizes.  Per-symbol returns
and transaction costs are tracked while a portfolio variance penalty discourages
excess risk.  Key parameters such as `rl_max_position`, `rl_transaction_cost`,
`rl_risk_penalty` and `rl_var_window` can be adjusted in `config.yaml`.  Set
`rl_algorithm: RLlib` together with `rllib_algorithm: PPO` or `DDPG` to train
using RLlib instead of Stable-Baselines.  The resulting checkpoint is stored
under `model_rllib/` and is automatically used by `generate_signals.py` when
`rl_algorithm` is set to `RLlib`.  Alternatively set `rl_algorithm: SAC` to
train a Soft Actor-Critic agent with Stable-Baselines3:

```yaml
rl_algorithm: SAC
rl_max_position: 1.0
rl_transaction_cost: 0.0001
rl_risk_penalty: 0.1
rl_var_window: 30
```
You can also specify `rl_algorithm: TRPO` to use the Trust Region Policy
Optimization implementation from `sb3-contrib`.  The trust region size is
controlled by `rl_max_kl` (default `0.01`).
`rl_algorithm: RecurrentPPO` trains the LSTM-based PPO implementation from
`sb3-contrib` and writes checkpoints to `models/recurrent_rl/`.
Example settings:

```yaml
rl_algorithm: RecurrentPPO
rl_steps: 10000
```
`rl_algorithm: HierarchicalPPO` enables the options framework from
`sb3-contrib` where a manager policy chooses trade direction and a worker
controls position sizing.  The trained model is stored as
`model_hierarchical.zip`.

```yaml
rl_algorithm: HierarchicalPPO
rl_steps: 10000
```
`rl_algorithm` can also be set to `A2C` or `A3C` to train Advantage Actor-Critic
agents with Stable-Baselines3.  `A3C` launches multiple parallel environments
with the number controlled by `rl_num_envs` (default `4`).
For a full pipeline combining all of these approaches run `train_combined.py`.

## Installation

1. Install Python dependencies:

   ```bash
   pip install -r requirements-core.txt
   # optional extras
   pip install .[heavy]
   pip install .[rl]
   pip install .[nlp]
   ```

   Dependencies are pinned for reproducibility. After verifying changes and tests pass, regenerate the lists with:

   ```bash
   pip freeze | sort > requirements-core.txt
   ```

2. Place historical CSV files under `data/`, specify a mapping of symbols to their download URLs in `config.yaml` under `data_urls`, **or** define `api_history` entries to fetch ticks directly from your MetaTrader&nbsp;5 terminal. Existing CSV files can be converted to Parquet using `python scripts/migrate_to_parquet.py`.
   The MT5 history center provides free tick data once you have logged in to a broker through the terminal. Programmatic access is also available via a helper that auto-selects symbols and handles chunked downloads:

   ```python
   from datetime import datetime
   from brokers.mt5_direct import fetch_history

   df = fetch_history("EURUSD", datetime(2024, 1, 1), datetime(2024, 1, 2))
   ```
3. The realtime trainer stores ticks in a DuckDB database located at `data/realtime.duckdb`. The database is created automatically the first time you run the script and old rows beyond the `realtime_window` setting are pruned on each update.
4. Set `use_feature_cache: true` in `config.yaml` to cache engineered features in `data/features.duckdb`. The cache is reused when the input history hasn't changed.

   Environment variables controlling the feature cache:

   - `FEATURE_CACHE_MAX_GB`: maximum size in gigabytes before least-recently-used items are evicted. Unset for no limit.
   - `FEATURE_CACHE_CODE_HASH`: optional code version string or hash included in the cache key. Changing it forces cache invalidation when feature code changes.

5. Optionally configure `feature_service_url` and `feature_service_api_key` to
   pull pre-computed features from a remote service over TLS. When the service
   cannot be reached the pipeline falls back to local feature engineering.
6. Adjust settings in `config.yaml` if needed. The `symbols` list controls which instruments are used for training.
7. Train the model and run a backtest:

   ```bash
   python train.py
   # run the transformer-based neural network
   python train_cli.py neural
   # train symbol-specific adapters
   python train_meta.py
   # train an AutoGluon TabularPredictor
   python train_autogluon.py
   python train_rl.py
   # end-to-end training of all components
   python train_combined.py
   python backtest.py
   ```

   The backtest now reports a bootstrap p-value for the Sharpe ratio:

   ```python
   from backtest import run_backtest
   from utils import load_config

   cfg = load_config("config.yaml")
   metrics = run_backtest(cfg)
   print(f"Sharpe: {metrics['sharpe']:.2f}, p-value: {metrics['sharpe_p_value']:.3f}")
   ```

   A small p-value (e.g. below 0.05) suggests the Sharpe ratio is unlikely to
   have occurred by chance.

   If `data_urls` are provided, `train.py` will download the file(s) for the configured symbols via `gdown` before training.
   When `api_history` entries are present, the data will instead be pulled directly from the MetaTrader&nbsp;5 history center.

To generate additional synthetic training sequences you can train either a GAN or diffusion model:

```bash
python scripts/train_tsgan.py        # TimeGAN based augmentation
python scripts/train_tsdiffusion.py  # Diffusion model (TimeGrad-style)
```

Enable `use_data_augmentation: true` to include the GAN samples or `use_diffusion_aug: true` to
blend in the diffusion sequences during model training. The number of diffusion training epochs
is controlled by the `diffusion_epochs` configuration key.

The resulting model file (`model.joblib`) can be loaded by the EA. When
training with `train_autogluon.py` the best predictor is stored under
`models/autogluon` and will be used when `model_type: autogluon` is set in
`config.yaml`.

To run live training and keep the repository in sync:

```bash
python realtime_train.py
```
This script continuously pulls ticks from the terminal, retrains the model and
pushes the updated dataset and model back to the repository.

```
python train_online.py
```
This optional script reads the features produced by `realtime_train.py` and
incrementally updates a lightweight river model stored under `models/online.joblib`.

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
   2. Run `pip install -r requirements-core.txt`. After verifying any updates, refresh the pinned versions with `pip freeze | sort > requirements-core.txt`.
   3. Install extras as needed, e.g. `pip install .[heavy]` or `pip install .[rl]`.
   4. For SHAP-based feature importance install `shap` with `pip install shap`.
      When the config option `feature_importance: true` is set, `train.py` and
      `train_cli.py neural` also write SHAP bar plots under `reports/` and a ranked
      `feature_importance.csv` file. For ad-hoc interpretation of an existing
      model run `python analysis/interpret_model.py` which produces a similar
      report for the most recent dataset.
   5. To enable the graph features plugin install `torch-geometric>=2.6`.
      This also enables optional GAT embeddings when `use_gat_features` is set.
      Set `use_temporal_gat` to use the sequential variant which also requires
      `torch-geometric`.
   6. To use the tsfresh features install `tsfresh` with `pip install tsfresh`.
   7. To use the FRED macro features install `pandas_datareader` with
      `pip install pandas_datareader` and set the `FRED_API_KEY` environment
      variable.
   8. The `auto_indicator` plugin automatically generates lag and rolling
      statistics for numeric columns and requires no extra dependencies.
      Lag lengths and window sizes can be configured and target columns
      can be skipped to avoid leakage.
7. **Build Protobuf classes** –
   1. Make sure the `protoc` compiler is installed and on your `PATH`.
   2. Run `protoc --python_out=. proto/signals.proto` from the repository root.
      This generates `proto/signals_pb2.py` which is imported by `signal_queue.py`.
8. **Initial training** –
   1. Still inside the command prompt run `python train.py`.
      The script downloads the backtesting files `XAUUSD.csv` and `GBPUSD.csv`
      from Google Drive and trains a LightGBM model.
   2. To experiment with the transformer-based neural network instead run `python train_cli.py neural`.
      This trains a small transformer on sequences of the same features and saves `model_transformer.pt`.
   3. After either script finishes you will see the resulting model file under the project folder.
   4. To browse logged runs start `scripts/mlflow_ui.sh` and open `http://localhost:5000` in your browser.

### Remote MLflow Tracking

To centralise experiment history, run an MLflow tracking server and point the
application at it via `config.yaml`.

**Docker**

```bash
docker run -d -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db \
  -e MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts \
  -v $(pwd)/mlflow:/mlflow \
  --name mlflow mlfloworg/mlflow:2.8.1 \
  mlflow server --host 0.0.0.0
```

**systemd**

```
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
ExecStart=/usr/local/bin/mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --default-artifact-root /var/mlflow
WorkingDirectory=/var/mlflow
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start with `systemctl enable --now mlflow`.

Configure the client in `config.yaml`:

```yaml
mlflow:
  tracking_uri: "http://your-server:5000"
  username: "secret://MLFLOW_USER"
  password: "secret://MLFLOW_PASS"
```
9. **Copy the EA** –
   1. Open MetaTrader 5 and click **File → Open Data Folder**.
   2. Run `python scripts/setup_terminal.py "<path-to-terminal>"` to automatically place `AdaptiveEA.mq5` and `RealtimeEA.mq5` inside `MQL5/Experts`.
   3. Restart MetaTrader 5 and compile the EA inside the MetaEditor by pressing **F7**.
10. **Attach the EA** –
   1. In MetaTrader 5 open the **Navigator** panel (Ctrl+N).
   2. Drag the EA onto a chart of either XAUUSD or GBPUSD and click **OK**.
11. **Publish signals** –
   1. In the command prompt run `python generate_signals.py`.
      This publishes prediction messages to `tcp://localhost:5555` which the EA subscribes to.
   2. Set the environment variable `SIGNAL_QUEUE_BIND` or `SIGNAL_QUEUE_URL` to change the port if needed.
   3. Messages are sent as Protobuf by default.  Set `SIGNAL_FORMAT=json` to
      publish plain JSON instead.  The Expert Advisors read from this queue via
      the `ZmqAddress` input and no longer require a `signals.csv` file.
   4. The script checks market hours using `exchange_calendars`; if the market
      is closed it runs a rolling backtest or falls back to historical data so
      analysis can continue. Pass `--simulate-closed-market` to manually test
      this behaviour.
12. **Run realtime training** –
   1. Back in the command prompt run `python realtime_train.py`.
   2. Leave this window open; the script will keep updating `model.joblib` as new ticks arrive.
13. **Optimise parameters** –
   1. Periodically run `python auto_optimize.py`.
      When `use_ray` is enabled the script leverages Ray Tune with
      `OptunaSearch` to distribute trials. Configure resource limits under
      `ray_resources` in `config.yaml`.
      The optimiser performs a Bayesian search across thresholds,
      walk‑forward window sizes and reinforcement‑learning parameters. Results
      are cross‑validated over multiple market regimes and both the metrics and
      chosen hyperparameters are tracked with MLflow. Any improvements are
      written back to `config.yaml` and logged under `logs/config_changes.csv`.
   2. To view experiment history run `scripts/mlflow_ui.sh` and open `http://localhost:5000`.
14. **Upload artifacts** –
   1. Start `python scripts/hourly_artifact_push.py` in a separate window. This
      script commits `logs/` and `checkpoints/` every hour so history is archived automatically. Use Windows Task
      Scheduler to launch it at logon for unattended operation.

15. **Keep it running** –
   1. Create scheduled tasks that start both `python realtime_train.py` and the
      hourly artifact uploader whenever the VPS boots or a user logs in. With these
      tasks enabled the bot and artifact push service run indefinitely.

With the EA running on your VPS and the training script collecting realtime data,
the bot will continually adapt to market conditions.

## MetaTrader 5 EA

The EA script `AdaptiveEA.mq5` demonstrates how to load predictions
produced by the Python model and place trades with a context aware trailing
stop. Trading no longer relies on MQL Expert Advisors; signals are transmitted
directly from Python to MetaTrader5 via the ``mt5_direct`` broker module.
This removes the external messaging bridge and simplifies deployment.

`generate_signals.py` merges ML probabilities with a moving average
crossover and RSI filter so trades are only taken when multiple conditions
confirm the direction.  Additional optional filters check for Bollinger band
breakouts, volume spikes and even macro indicators when a `macro.csv` file is
present. Configuration values for these filters live in `config.yaml`.  The
pipeline now also considers news sentiment scores and cross-asset momentum
to further refine entries. Set `enable_news_trading` to `false` to automatically
block trades within a few minutes of scheduled high impact events pulled from
all three calendars.  When the optional `transformers` dependency is available,
news summaries are encoded with a HuggingFace
`AutoModelForSequenceClassification` to produce both polarity scores and
embedding vectors that downstream models can consume.  If the model cannot be
loaded the features gracefully fall back to zeros so existing pipelines remain
operational.

Sentiment scoring behaviour can be tuned via `sentiment_mode` in
`config.yaml`. The default `full` mode loads the original FinBERT/FinGPT
models, `lite` uses a small distilled model for faster local inference and
`remote` posts texts to the URL specified by `sentiment_api_url` expecting a
JSON response with a `scores` array (and optional `summaries` for FinGPT).

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
be slow. The script `train_parallel.py` and the hyper‑parameter optimiser use
Ray to distribute work across CPU cores or even a cluster. Enable this by
setting `use_ray: true` in `config.yaml` and optionally adjust the new
`ray_resources` section to control the CPUs and GPUs allocated per trial.
Start a cluster with:

```bash
ray start --head
# on each worker
ray start --address='<head-ip>:6379'
```

Once running, any call to `python auto_optimize.py` or `train_parallel.py` will
automatically utilise the cluster.
The Docker setup remains unchanged so experiments stay reproducible.

Feature engineering in `dataset.make_features` can also leverage Dask for
out-of-core processing. Set `use_dask: true` and optionally provide
`dask_cluster_url` pointing to a running scheduler. Without a URL a local
cluster is launched automatically. Install `dask[distributed]` and start workers
with:

```bash
dask-scheduler
dask-worker tcp://<scheduler-ip>:8786
```

## Plugin Architecture

Feature engineering functions, models and risk checks can now be extended via
plugins under the `plugins/` package. Register new components with the helper
decorators exposed in `plugins.__init__` and they will automatically be applied
when `dataset.make_features` or training scripts run.
Built-in examples include the `atr`, `donchian` and `keltner` plugins which add
ATR, Donchian and Keltner channel signals. A regime classification plugin can
also be enabled to label each row using a hidden Markov model.
Risk checks for spread limits and slippage detection are provided by the
`spread` and `slippage` modules.  A reinforcement learning based sizing policy
is implemented in `plugins/rl_risk.py`; it outputs a position size multiplier
given recent returns and risk metrics.  The policy is trained by
`train_rl.py` and saved under `models/` for use during live trading.

## Strategy Templates

Example strategies adapted from the open source Freqtrade and Backtrader
frameworks are included under `strategies/`. They demonstrate how the feature
pipeline can feed different trading styles ranging from simple MA crossovers to
Donchian breakouts with ATR based stops.

## External Strategy Integration

The backtester can also execute strategies written for Freqtrade or Backtrader.
Install the optional dependencies:

```bash
pip install backtrader freqtrade ta-lib
```

Run a strategy using the new CLI flag:

```bash
python backtest.py --external-strategy strategies/freqtrade_template.py
```

The adapter automatically detects the framework and feeds the existing feature
dataframe to the strategy, recording the resulting trades.

## Detailed Logging

All scripts now emit structured JSON logs to `logs/app.log` with rotation to
prevent the file from growing indefinitely. The `log_utils` module also patches
`print` so anything printed to the console is captured in the log file. Key
functions are wrapped with a decorator to record start/end markers and any
exceptions. Each JSON record includes timestamp, log level and module name.

These logs can be ingested directly by monitoring stacks such as the ELK stack
or Grafana Loki by enabling JSON parsing on the collector. For quick
inspection, pipe the log through `jq`:

```bash
tail -f logs/app.log | jq .
```

The helper script `scripts/sync_artifacts.py` can be run to automatically commit and push the log and checkpoint directories to your repository for later analysis.

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
`train_cli.py neural` whenever new data is pushed or on a daily schedule. Generated
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
can be overridden via environment variables defined in the deployment manifest.

## Remote Management API

A small FastAPI application defined in `remote_api.py` exposes REST endpoints for
starting and stopping multiple bots. Launch the server with TLS enabled:

```bash
python remote_api.py
```

The process reads `certs/api.crt` and `certs/api.key` and **fails to start** if
`API_KEY` is unset. Include the header `X-API-Key` in each request. Key
endpoints include:

- `GET /bots` – list running bot IDs and whether each is alive.
- `POST /bots/<id>/start` – launch a new training process.
- `POST /bots/<id>/stop` – terminate an existing process.
- `GET /bots/<id>/status` – return the bot's PID, exit code and a log tail.
- `GET /bots/<id>/logs` – fetch recent log lines for the bot.
- `GET /logs` – fetch the last few lines from `logs/app.log`.
- `GET /health` – overall service state and recent log snippet.
- `POST /config` – update `config.yaml` with JSON fields `key`, `value` and
  `reason`.
- `GET /risk/status` – aggregated exposure, daily loss, VaR and whether trading
  has been halted by the risk manager.

Example:

```bash
curl -k -H "X-API-Key: <token>" https://localhost:8000/health
```

Set `MAX_PORTFOLIO_DRAWDOWN` (and optional `MAX_VAR`) before launching
`remote_api.py` to enable the portfolio level risk manager:

```bash
MAX_PORTFOLIO_DRAWDOWN=1000 python remote_api.py
```

### API Key Rotation

Generate a new key and update the `API_KEY` environment variable or Kubernetes
secret. Restart the service or roll your deployment so pods pick up the new
value. Once clients have been updated to use the new token, remove the old key
from the secret or environment.

### Secret Provisioning

Configuration values that contain sensitive information now reference secrets
using the `secret://` scheme. At runtime these references are resolved by
`utils.SecretManager`, which first checks environment variables and then (if
configured) a Vault-compatible backend. To provide secrets:

1. Populate the required environment variables or store the values in Vault.
2. Update `config.yaml` with `secret://<NAME>` entries, for example
   `alerting.slack_webhook: "secret://SLACK_WEBHOOK"`.
3. Ensure unit tests patch `SecretManager.get_secret` when they require specific
   values.

This approach keeps credentials out of source control while still allowing
local development without Vault by simply setting environment variables.

### Monitoring and Dashboard

The API now exposes Prometheus metrics at `/metrics`. Deploying Prometheus and
Grafana lets you monitor CPU/RAM usage, queue depth, trade counts and drift events
in real time. Add the following scrape annotations to the deployment so
Prometheus discovers the service:

```yaml
    annotations:
      prometheus.io/scrape: "true"
      prometheus.io/path: "/metrics"
      prometheus.io/port: "8000"
      prometheus.io/scheme: "https"
```

Import the provided Grafana dashboard JSON in the `grafana/` folder to visualise
these metrics. Sample Prometheus and alerting configurations are available in
`deploy/`. See `docs/monitoring.md` for details on enabling optional Alertmanager
rules.

**Note:** Keep this section updated whenever deployment scripts or automation
change to avoid configuration drift.

### Data Drift Mitigation

When monitoring detects significant drift between live data and training
distributions, consider the following responses:

* **Retrain** – update models on recent data so they adapt to new market
  conditions.
* **Rollback** – revert to a previously stable model version while the cause of
  drift is investigated.

---

This is a simplified template intended for further extension and tuning.  It is
up to the user to verify performance using additional backtests and forward
testing within MetaTrader 5 before deploying to a live environment.

## Meta-learning with MAML

`meta_train_nn.py` trains the transformer model across multiple symbols using
Model-Agnostic Meta-Learning (MAML). The resulting initialisation is saved to
`models/meta_transformer.pth`. Enable it by setting `use_meta_model: true` in
`config.yaml`; `generate_signals.py` will load the weights and blend the
meta-model's probabilities with any other ensemble members. During evaluation on
an unseen symbol the script fine-tunes the meta-trained weights on a small
subset before producing signals, controlled by `finetune_steps`.

Key configuration options:

* `meta_epochs` – outer-loop epochs.
* `meta_batch_size` – number of symbols per meta-batch.
* `inner_lr` – inner-loop learning rate.
* `inner_steps` – gradient steps during adaptation.
* `finetune_steps` – steps used when adapting to a new symbol.
* `use_meta_model` – include the meta-trained transformer during signal generation.

A tiny example dataset is provided at `tests/sample_meta.csv` for quick
testing of the meta-training workflow.

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

The live signal generator can also blend in predictions from an incremental
model trained with `train_online.py`. Set `use_online_model: true` in
`config.yaml` and run the trainer alongside `realtime_train.py` to enable this
behaviour.

## Mixed Precision and Checkpointing

The neural subcommand supports memory-saving options controlled by two flags in
`config.yaml`:

* `use_amp` – enables automatic mixed precision, cutting GPU memory use and
  often improving training throughput.
* `use_checkpointing` – checkpoints transformer encoder layers, reducing peak
  memory at the cost of extra compute.

These options allow larger models or batch sizes to fit in memory; while AMP
can speed up training, checkpointing may slow it slightly due to recomputation.

## Automatic Log Uploading

`scripts/sync_artifacts.py` commits the contents of `logs/`, `checkpoints/` and `config.yaml` to
the repository. Set the `GITHUB_TOKEN` environment variable to a token with
write access before running:

```bash
GITHUB_TOKEN=<token> python scripts/sync_artifacts.py
```

Long-running services can import and call
`scripts.sync_artifacts.register_shutdown_hook()` to push artifacts when they exit. To
upload logs periodically, schedule the script with cron, for example:

```cron
0 * * * * GITHUB_TOKEN=<token> /usr/bin/python /path/to/scripts/sync_artifacts.py
```

## Docker image

The `docker/Dockerfile` builds a reproducible environment with core
dependencies. GPU extras can be enabled through build arguments:

```bash
# CPU-only image
docker build -f docker/Dockerfile -t mt5:latest .

# GPU-enabled image
docker build \
  -f docker/Dockerfile \
  --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04 \
  --build-arg INSTALL_GPU_EXTRAS=true \
  -t mt5:gpu .
```

CI publishes this image to GitHub Container Registry so it can be pulled
directly. The container's entrypoint automatically runs a resource monitor and
enables any plugins appropriate for the detected hardware tier. Starting an
interactive Python shell with the preconfigured environment is therefore as
simple as:

```bash
docker run --rm ghcr.io/OWNER/mt5:latest
```

From there you can invoke project modules, for example:

```bash
docker run --rm ghcr.io/OWNER/mt5:latest python -m utils.environment
```

The `docker-compose.yml` file launches the remote API, Streamlit dashboard and
risk manager services with `./checkpoints` and `./logs` mounted for persistent
storage:

```bash
docker-compose up --build
```

## gRPC Management API

The gRPC management API is served over TLS. Place the server certificate
and key at `certs/server.crt` and `certs/server.key` respectively. Clients
must provide the CA certificate `certs/ca.crt` when establishing a
connection. The example client in `scripts/grpc_client.py` illustrates how
to configure a secure channel.

When deploying, mount these certificate files into the `certs/` directory
so the service can load them on startup.

## Debian package installation

Download the prebuilt .deb from releases and install:

```bash
sudo dpkg -i mt5bot_<ver>.deb
sudo systemctl start mt5bot
sudo systemctl status mt5bot
sudo systemctl stop mt5bot
```

## Quality Checks

Run the placeholder check before committing or tagging a release:

```bash
python scripts/check_skeletons.py
```

The script scans the repository for stray `pass` statements, `TODO` markers and
`NotImplementedError` usage outside of approved locations. It runs as part of
CI and exits with a non‑zero status when skeleton code is found.

## License

This project is licensed under the [MIT License](LICENSE).
