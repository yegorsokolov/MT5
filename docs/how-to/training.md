# Training Guide

This guide shows how to launch a basic training run.

```bash
python -m mt5.train --config config.yaml
```

```{doctest}
>>> 1 + 1
2
```

## Walk-forward training

Run sequential train/test windows and log metrics to MLflow using the
``walk-forward`` subcommand:

```bash
python -m mt5.train_cli walk-forward --data prices.csv --window-length 100 --step-size 10 --model-type mean
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

``mt5.train_parallel`` and ``mt5.train_nn`` derive candidate feature columns
directly from the dataframe returned by :func:`data.features.make_features`.
The training CLI now seeds this candidate list using
``training.feature_includes`` (with sensible defaults when unspecified), prunes
entries with ``training.feature_excludes`` and evaluates the
``training.feature_families`` toggles before invoking
:func:`analysis.feature_selector.select_features`.  You can also define your own
named bundles via ``training.feature_groups`` and reference them from
``feature_families``.

```yaml
training:
  feature_families:
    baseline: true   # always keep baseline signals & stops
    order_flow: true # retain imbalance/CVD features
    cross_spectral: false
    macro: true
  feature_includes:
    - risk_tolerance
  feature_excludes:
    - noise_feature
  feature_groups:
    macro:
      - macro_spread
```

This makes richer signals—baseline strategy confidence, order-flow imbalance
and cross-spectral coherence—available to the optimiser by default, while
letting you opt into entirely custom feature packs without touching the code.
