"""Key hashing, key parsing, and provider-key wrapping."""

import os

from ragengine.crypto import (
    decrypt_secret,
    encrypt_secret,
    generate_tenant_key,
    hash_key,
    slug_from_key,
    verify_key,
)
from ragengine.config import get_settings
from ragengine.exceptions import ConfigurationError

from .base import EngineTestCase


class TenantKeyTests(EngineTestCase):
    def test_key_carries_slug_and_verifies(self):
        key = generate_tenant_key("acme")
        self.assertEqual(slug_from_key(key), "acme")
        self.assertTrue(verify_key(key, hash_key(key)))

    def test_wrong_key_fails_verification(self):
        key = generate_tenant_key("acme")
        other = generate_tenant_key("acme")
        self.assertFalse(verify_key(other, hash_key(key)))

    def test_malformed_keys_have_no_slug(self):
        for bad in ("", "erag", "erag.acme", "nope.acme.secret", "erag..secret"):
            self.assertIsNone(slug_from_key(bad), bad)


class SecretWrappingTests(EngineTestCase):
    def test_plain_scheme_roundtrip_without_secret_key(self):
        payload = encrypt_secret("sk-ant-something")
        self.assertEqual(payload["scheme"], "plain")
        self.assertEqual(decrypt_secret(payload), "sk-ant-something")

    def test_fernet_used_when_available(self):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            self.skipTest("cryptography not installed")
        os.environ["ENTERPRISE_SECRET_KEY"] = "operator-secret"
        get_settings.cache_clear()
        payload = encrypt_secret("sk-ant-something")
        self.assertEqual(payload["scheme"], "fernet")
        self.assertNotIn("sk-ant-something", payload["value"])
        self.assertEqual(decrypt_secret(payload), "sk-ant-something")

    def test_unknown_scheme_is_refused(self):
        with self.assertRaises(ConfigurationError):
            decrypt_secret({"scheme": "rot13", "value": "x"})

    def test_fernet_payload_without_key_is_refused(self):
        with self.assertRaises(ConfigurationError):
            decrypt_secret({"scheme": "fernet", "value": "gAAAA-not-decryptable"})
