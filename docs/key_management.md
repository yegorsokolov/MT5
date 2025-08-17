# Key Management

Sensitive artifacts such as model checkpoints and decision logs are encrypted
at rest using AES-256-GCM.  Keys are resolved via `utils.SecretManager`, which
looks up values from environment variables or HashiCorp Vault.

## Configuration

- `CHECKPOINT_AES_KEY` – base64 encoded 32 byte key used to encrypt
  `checkpoints/checkpoint_*.pkl.enc`.
- `DECISION_AES_KEY` – base64 encoded 32 byte key for
  `logs/decisions.parquet.enc`.

Keys should be rotated periodically.  The provided `scripts/rotate_keys.py`
utility re-encrypts existing artifacts with new keys retrieved from the
`SecretManager`.

## Backup and Restore

Encrypted files are backed up transparently by `core.BackupManager` and
replicated by `core.state_sync`.  To restore, ensure the appropriate keys are
available in the environment or Vault before loading checkpoints or decision
logs.
