# Training Guide

This guide shows how to launch a basic training run.

```bash
python train.py --config config.yaml
```

```{doctest}
>>> 1 + 1
2
```

## Walk-forward training

Run sequential train/test windows and log metrics to MLflow using the
``walk-forward`` subcommand:

```bash
python train_cli.py walk-forward --data prices.csv --window-length 100 --step-size 10 --model-type mean
```

Each window trains on the preceding ``window-length`` rows and evaluates on
the following ``step-size`` rows, avoiding data leakage.

## Handling uneven timestamps

Transformers in the toolkit can learn about irregular sampling intervals by
enabling the ``time_encoding`` flag in ``config.yaml``. When set to ``true`` the
models inject :class:`~models.time_encoding.TimeEncoding` features before every
attention block so sequences with uneven spacing no longer appear identical to
the network. This typically reduces validation loss on feeds where gaps between
bars vary, such as event-driven tick streams.
