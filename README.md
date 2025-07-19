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
   to better match current market conditions.
3. **Realtime trainer** — a Python script that fetches live ticks from MT5,
   incrementally retrains the model and commits updates to this GitHub repo.

The feature engineering step now includes additional indicators such as
lower/higher timeframe moving averages, a volatility measure and basic
order-book statistics (spread and volume imbalance). These richer features help
the model capture more market behaviour than simple MAs and RSI alone.

The project can be adapted to any symbol by changing the configuration
parameters and retraining the model on the corresponding historical data.
`train.py` now supports training on multiple symbols at once.  By default both
`XAUUSD` and `GBPUSD` history files will be downloaded and combined.
An alternative script `train_nn.py` trains a small LSTM network on sliding
windows of these features for those wanting to explore deep learning models.
`train_meta.py` demonstrates a simple meta-learning approach where a global
model is fitted on all symbols and lightweight adapters are fine-tuned for each
instrument.  The per-symbol models are saved under the `models/` folder.
Another option `train_rl.py` trains a reinforcement learning agent that optimises risk-adjusted profit.

## Installation

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place historical CSV files under `data/` **or** specify a mapping of symbols to their download URLs in `config.yaml` under `data_urls`.
3. Adjust settings in `config.yaml` if needed. The `symbols` list controls which instruments are used for training.
4. Train the model and run a backtest:

   ```bash
   python train.py
   # or use the experimental neural network
   python train_nn.py
   # train symbol-specific adapters
   python train_meta.py
   python train_rl.py
   python backtest.py
   ```

   If `data_urls` are provided, `train.py` will download the file(s) for the configured symbols via `gdown` before training.

The resulting model file (`model.joblib`) can be loaded by the EA.

To run live training and keep the repository in sync:

```bash
python realtime_train.py
```
This script continuously pulls ticks from the terminal, retrains the model and
pushes the updated dataset and model back to the repository.

## Deployment Guide

Follow these steps to run the EA and the realtime trainer on a Windows VPS:

1. **Provision a VPS** – Create a Windows VPS with at least 2 GB of RAM.
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
      The script downloads the XAUUSD and GBPUSD history files and trains a LightGBM model.
   2. To experiment with a recurrent neural network instead run `python train_nn.py`.
      This trains a small LSTM on sequences of the same features and saves `model_lstm.pt`.
   3. After either script finishes you will see the resulting model file under the project folder.
7. **Copy the EA** –
   1. Open MetaTrader 5 and click **File → Open Data Folder**.
   2. Navigate to `MQL5/Experts` and copy `AdaptiveEA.mq5` (or `RealtimeEA.mq5`) into this directory.
   3. Restart MetaTrader 5 and compile the EA inside the MetaEditor by pressing **F7**.
8. **Attach the EA** –
   1. In MetaTrader 5 open the **Navigator** panel (Ctrl+N).
   2. Drag the EA onto a chart of either XAUUSD or GBPUSD and click **OK**.
9. **Run realtime training** –
   1. Back in the command prompt run `python realtime_train.py`.
   2. Leave this window open; the script will keep updating `model.joblib` as new ticks arrive.

With the EA running on your VPS and the training script collecting realtime data,
the bot will continually adapt to market conditions.

## MetaTrader 5 EA

The EA script `AdaptiveEA.mq5` demonstrates how to load predictions
produced by the Python model and place trades with a context aware trailing
stop.  `RealtimeEA.mq5` extends this idea by automatically running `git pull`
on initialisation so the latest model from GitHub is used.

`generate_signals.py` merges ML probabilities with a moving average
crossover and RSI filter so trades are only taken when multiple conditions
confirm the direction.  Additional optional filters check for Bollinger band
breakouts, volume spikes and even macro indicators when a `macro.csv` file is
present. Configuration values for these filters live in `config.yaml`.

## Performance Reports

`backtest.py` outputs statistics including win rate, Sharpe ratio and maximum
drawdown. These metrics can be used to iteratively optimise the strategy.

---

This is a simplified template intended for further extension and tuning.  It is
up to the user to verify performance using additional backtests and forward
testing within MetaTrader 5 before deploying to a live environment.
