"""How secrets are protected: tenant keys are hashed, provider keys encrypted.

Two different problems, two different treatments:

* A **tenant API key** only needs to be *verified*, never read back — so only
  a SHA-256 digest is stored and the plaintext is shown exactly once, at
  creation. A database dump cannot reveal tenant keys.

* A **provider key** (the tenant's Anthropic or Gemini credential) must be
  *used* on every request, so it has to be recoverable. It is encrypted with
  Fernet under ``ENTERPRISE_SECRET_KEY`` when the ``cryptography`` package is
  installed, and stored plainly (file mode 0600) otherwise — the scheme is
  recorded next to the value so a deployment can be upgraded in place.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Optional

from .config import get_settings
from .exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Tenant API keys: generate, hash, verify.
# ---------------------------------------------------------------------------


def generate_tenant_key(slug: str) -> str:
    """Mint a tenant API key.

    The slug rides along in the key (``erag.<slug>.<secret>``) so lookup is a
    dictionary access rather than a scan; all of the entropy — and all of the
    security — is in the final segment.
    """
    return f"erag.{slug}.{secrets.token_urlsafe(24)}"


def hash_key(api_key: str) -> str:
    """The digest that gets stored in place of the key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_key(api_key: str, stored_hash: str) -> bool:
    """Constant-time comparison so keys can't be guessed a byte at a time."""
    return hmac.compare_digest(hash_key(api_key), stored_hash)


def slug_from_key(api_key: str) -> Optional[str]:
    """Pull the tenant slug out of a presented key, or None if malformed."""
    parts = api_key.split(".")
    if len(parts) != 3 or parts[0] != "erag" or not parts[1] or not parts[2]:
        return None
    return parts[1]


# ---------------------------------------------------------------------------
# Provider keys: encrypt at rest, decrypt on use.
# ---------------------------------------------------------------------------


def _fernet():
    """A Fernet instance derived from ENTERPRISE_SECRET_KEY, or None.

    Returns None when either the ``cryptography`` package or the secret is
    missing, so callers can fall back rather than refuse to run. The Fernet
    key is derived by hashing, so the operator secret can be any string.
    """
    secret = get_settings().secret_key
    if not secret:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(value: str) -> dict:
    """Wrap a provider key for storage, recording which scheme was used."""
    fernet = _fernet()
    if fernet is not None:
        return {"scheme": "fernet", "value": fernet.encrypt(value.encode("utf-8")).decode("ascii")}
    # Deliberately explicit rather than silently insecure: the scheme marker
    # makes "this deployment stores keys unencrypted" visible in the file.
    return {"scheme": "plain", "value": value}


def decrypt_secret(payload: dict) -> str:
    """Unwrap a stored provider key."""
    scheme, value = payload.get("scheme"), payload.get("value", "")
    if scheme == "plain":
        return value
    if scheme == "fernet":
        fernet = _fernet()
        if fernet is None:
            raise ConfigurationError(
                "A provider key is Fernet-encrypted but ENTERPRISE_SECRET_KEY "
                "(or the cryptography package) is missing in this environment."
            )
        from cryptography.fernet import InvalidToken

        try:
            return fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise ConfigurationError(
                "Could not decrypt a provider key — ENTERPRISE_SECRET_KEY has "
                "changed since the tenant was created."
            ) from exc
    raise ConfigurationError(f"Unknown secret scheme {scheme!r}")
