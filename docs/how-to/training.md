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

## Feature selection and families

``train_parallel.py`` and ``train_nn.py`` now derive candidate feature columns
directly from the dataframe returned by :func:`data.features.make_features`.
All numeric columns except identifiers (``Timestamp``, ``Symbol``) and labels
are passed through :func:`analysis.feature_selector.select_features`.  Entire
feature families such as ``baseline``, ``order_flow`` and ``cross_spectral`` can
be retained or dropped via configuration without touching code:

```yaml
training:
  feature_families:
    baseline: true   # always keep baseline signals & stops
    order_flow: true # retain imbalance/CVD features
    cross_spectral: false
  feature_includes:
    - risk_tolerance
  feature_excludes:
    - noise_feature
```

This makes richer signals—baseline strategy confidence, order-flow imbalance
and cross-spectral coherence—available to the optimiser by default, improving
validation metrics on the bundled datasets.
