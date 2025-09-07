from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from utils import load_config
from data.history import load_history_config
from data.features import make_features, make_sequence_arrays
from models.ts_masked_encoder import train_ts_masked_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train masked time-series encoder")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument(
        "--store-dir", type=Path, default=None, help="Optional directory for model_store"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override training batch size"
    )
    args = parser.parse_args()

    if args.config is not None:
        os.environ["CONFIG_FILE"] = str(args.config)
    cfg = load_config()

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    if not symbols:
        raise SystemExit("No symbols configured")
    sym = symbols[0]
    df = load_history_config(sym, cfg, Path.cwd(), validate=cfg.get("validate", False))
    feat_df = make_features(df, cfg.get("features", []))
    seq_len = cfg.get("seq_len", 16)
    windows, _ = make_sequence_arrays(feat_df, feat_df.columns.tolist(), seq_len)
    data = torch.tensor(windows, dtype=torch.float32)

    epochs = args.epochs or cfg.get("ts_pretrain_epochs", 20)
    batch_size = args.batch_size or cfg.get("ts_pretrain_batch_size", 32)
    train_ts_masked_encoder(data, epochs=epochs, batch_size=batch_size, store_dir=args.store_dir)


if __name__ == "__main__":
    main()
