# Adaptive MT5 Trading Bot

This folder contains a modular Expert Advisor (EA) and supporting Python scripts for training and
backtesting machine learning driven trading strategies on MetaTrader 5.

## Overview

The solution is split into two components:

1. **Python toolkit** — handles data preprocessing, feature engineering, machine
   learning and backtesting using the provided CSV history files.
2. **MQL5 Expert Advisor** — a lightweight EA that loads model signals and
   executes trades with a dynamic trailing stop while respecting FTMO style risk
   limits.
3. **Realtime trainer** — a Python script that fetches live ticks from MT5,
   incrementally retrains the model and commits updates to this GitHub repo.

The project can be adapted to any symbol by changing the configuration
parameters and retraining the model on the corresponding historical data.

## Installation

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place historical CSV files under `data/`.
3. Adjust settings in `config.yaml` if needed.
4. Train the model and run a backtest:

   ```bash
   python train.py
   python backtest.py
   ```

The resulting model file (`model.joblib`) can be loaded by the EA.

To run live training and keep the repository in sync:

```bash
python realtime_train.py
```
This script continuously pulls ticks from the terminal, retrains the model and
pushes the updated dataset and model back to the repository.

## MetaTrader 5 EA

The EA script `AdaptiveEA.mq5` demonstrates how to load predictions
produced by the Python model and place trades with a context aware trailing
stop.  `RealtimeEA.mq5` extends this idea by automatically running `git pull`
on initialisation so the latest model from GitHub is used.

## Performance Reports

`backtest.py` outputs statistics including win rate, Sharpe ratio and maximum
drawdown. These metrics can be used to iteratively optimise the strategy.

---

This is a simplified template intended for further extension and tuning.  It is
up to the user to verify performance using additional backtests and forward
testing within MetaTrader 5 before deploying to a live environment.
