from __future__ import annotations

import argparse
from pathlib import Path
from mt5.crypto_utils import encrypt, decrypt, _load_key


def rotate_keys(
    checkpoint_old: str,
    checkpoint_new: str,
    decision_old: str,
    decision_new: str,
    checkpoint_dir: Path = Path("checkpoints"),
    decision_path: Path = Path("logs/decisions.parquet.enc"),
) -> None:
    """Re-encrypt checkpoints and decision logs with new AES keys."""
    old_ck = _load_key(checkpoint_old)
    new_ck = _load_key(checkpoint_new)
    for ckpt in checkpoint_dir.glob("checkpoint_*.pkl.enc"):
        data = decrypt(ckpt.read_bytes(), old_ck)
        ckpt.write_bytes(encrypt(data, new_ck))
    if decision_path.exists():
        old_dl = _load_key(decision_old)
        new_dl = _load_key(decision_new)
        data = decrypt(decision_path.read_bytes(), old_dl)
        decision_path.write_bytes(encrypt(data, new_dl))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate encryption keys")
    parser.add_argument("--checkpoint-old", required=True, help="Old checkpoint key ref")
    parser.add_argument("--checkpoint-new", required=True, help="New checkpoint key ref")
    parser.add_argument("--decision-old", required=True, help="Old decision log key ref")
    parser.add_argument("--decision-new", required=True, help="New decision log key ref")
    args = parser.parse_args()
    rotate_keys(
        args.checkpoint_old,
        args.checkpoint_new,
        args.decision_old,
        args.decision_new,
    )


if __name__ == "__main__":
    main()
