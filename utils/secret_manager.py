import os
from typing import Optional

try:
    import hvac  # type: ignore
except Exception:  # pragma: no cover - hvac optional
    hvac = None  # type: ignore


class SecretManager:
    """Resolve secrets from environment variables or Vault.

    Secrets are referenced using the ``secret://`` scheme. The portion
    after the scheme is treated as the secret name. Resolution attempts
    the following in order:

    1. Environment variables.
    2. HashiCorp Vault KV v2 backend when ``VAULT_ADDR`` and
       ``VAULT_TOKEN`` are configured and the ``hvac`` library is
       available.
    """

    def __init__(self, vault_addr: Optional[str] = None, token: Optional[str] = None) -> None:
        self.vault_addr = vault_addr or os.getenv("VAULT_ADDR")
        self.token = token or os.getenv("VAULT_TOKEN")
        self._client = None
        if self.vault_addr and self.token and hvac:
            try:
                self._client = hvac.Client(url=self.vault_addr, token=self.token)
            except Exception:  # pragma: no cover - connection issues
                self._client = None

    # ------------------------------------------------------------------
    def get_secret(self, ref: str, default: Optional[str] = None) -> Optional[str]:
        """Return the secret value for ``ref``.

        ``ref`` may optionally start with ``secret://``. If a value cannot
        be resolved, ``default`` is returned instead of raising an
        exception.
        """

        key = ref.split("secret://", 1)[-1]

        # Environment variable lookup
        env = os.getenv(key)
        if env is not None:
            return env

        # Vault lookup
        if self._client:
            try:  # pragma: no cover - requires external service
                secret = self._client.secrets.kv.v2.read_secret_version(path=key)
                data = secret.get("data", {}).get("data", {})
                # attempt to return value stored under "value" or key name
                return data.get("value") or data.get(key)
            except Exception:
                pass

        return default
