# Online Training Updates

To keep models responsive to market changes, schedule periodic online updates.

- Run `dvc pull` before training to sync the latest raw/history data.
- After generating new data, run `dvc add` and `dvc push` to version it.

- **LightGBM**: run `python -m mt5.train --resume-online` whenever new feature data is
  available. Training state is saved after each mini‑batch using LightGBM's
  `init_model` support.
- **Neural network**: run `python -m mt5.train_cli neural --resume-online` to continue
  training on fresh sequences. Optimizer state and parameters are persisted after
  every mini‑batch.

A simple cron entry can refresh models hourly:

```cron
0 * * * * cd /workspace/MT5 && /usr/bin/python -m mt5.train --resume-online
0 * * * * cd /workspace/MT5 && /usr/bin/python -m mt5.train_cli neural --resume-online
```

Adjust frequency to match data latency and infrastructure capacity.
