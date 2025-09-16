# Feature Benchmarks

This note summarizes the observed impact of the latest feature-engineering
optimizations and how to inspect the associated metadata.

## Vectorized sequence generation

The vectorized implementations for `train_test_split` and
`make_sequence_arrays` remove the symbol-by-symbol Python loops. A benchmark on
a dataframe with 20 symbols and 5,000 timestamps produced the following
wall-clock times (seconds):

| Function                 | Legacy | Vectorized |
|--------------------------|--------|------------|
| `train_test_split`       | 0.0650 | 0.0463     |
| `make_sequence_arrays`   | 0.4167 | 0.1060     |

These figures correspond to the run captured in the benchmark output and show
a 1.4× speed-up for the split helper and a 3.9× reduction for sequence
materialisation.【a7191c†L1-L1】

## Cross-asset dimensionality control

The cross-asset feature generator now records a compact summary of the column
budget in `df.attrs["cross_asset_feature_summary"]` together with the current
`max_pairs` limit. Enabling `max_pairs=50` with `reduce="top_k"` constrains the
pairwise surface to 200 columns, providing predictable model widths even when
hundreds of unique symbol pairs are possible.【13d6a6†L1-L3】 The summary also
captures the active reduction strategy, unique pair count, and how many
relative-strength columns were produced.【F:features/cross_asset.py†L423-L431】

## Inspecting per-feature runtimes

`make_features` records per-module timings and the number of newly introduced
columns in `df.attrs["feature_timings"]`. Each entry includes `start`, `end`,
`duration`, and `columns`, enabling downstream diagnostics and alerting.【F:data/features.py†L669-L704】【F:data/features.py†L723-L737】【F:data/features.py†L1000-L1005】

