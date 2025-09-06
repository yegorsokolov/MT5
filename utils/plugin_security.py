from __future__ import annotations

"""Utilities for signing and verifying plugin modules.

Plugins are signed using RSA public key cryptography with SHA256 hashes.  Each
plugin module ships with a ``.sig`` file containing the signature of the module
source code.  Signatures are verified prior to plugin registration.
"""

from pathlib import Path
from typing import Optional

import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Location of the public key used for signature verification.  The private key
# is held by maintainers and not distributed with the repository.
PUBLIC_KEY_PATH = Path(__file__).resolve().parents[1] / "certs" / "plugin_pubkey.pem"


def load_public_key(path: Path = PUBLIC_KEY_PATH):
    """Load the RSA public key from ``path``."""
    data = path.read_bytes()
    return serialization.load_pem_public_key(data)


def verify_plugin(
    path: Path, sig_path: Optional[Path] = None, public_key_path: Path | None = None
) -> None:
    """Verify ``path`` against its signature.

    Parameters
    ----------
    path: Path
        Path to the plugin module to verify.
    sig_path: Optional[Path]
        Optional path to the signature file.  Defaults to ``path`` with a
        ``.sig`` extension appended.
    public_key_path: Optional[Path]
        Optional path to the public key.  Defaults to :data:`PUBLIC_KEY_PATH`.

    Raises
    ------
    FileNotFoundError
        If the signature file is missing.
    cryptography.exceptions.InvalidSignature
        If the signature check fails.
    """

    sig_path = sig_path or path.with_suffix(path.suffix + ".sig")
    pub_path = public_key_path or PUBLIC_KEY_PATH

    if not sig_path.exists():
        raise FileNotFoundError(f"Missing signature file {sig_path}")

    public_key = load_public_key(pub_path)
    data = path.read_bytes()
    sig_b64 = sig_path.read_text().strip()
    signature = base64.b64decode(sig_b64)
    public_key.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())


def sign_plugin(
    path: Path, private_key_path: Path, sig_path: Optional[Path] = None
) -> bytes:
    """Sign ``path`` using the private key and write the ``.sig`` file."""
    sig_path = sig_path or path.with_suffix(path.suffix + ".sig")
    private_key = serialization.load_pem_private_key(
        private_key_path.read_bytes(), password=None
    )
    data = path.read_bytes()
    signature = private_key.sign(data, padding.PKCS1v15(), hashes.SHA256())
    sig_path.write_text(base64.b64encode(signature).decode("ascii") + "\n")
    return signature


__all__ = ["verify_plugin", "sign_plugin", "load_public_key", "PUBLIC_KEY_PATH"]
