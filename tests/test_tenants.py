"""The tenant registry: onboarding, lookup, authentication, isolation."""

import json

from ragengine.exceptions import (
    AuthenticationError,
    ConfigurationError,
    TenantExistsError,
    TenantNotFoundError,
)
from ragengine.tenants import DEFAULT_MODELS, get_tenant_store, slugify

from .base import EngineTestCase


class SlugifyTests(EngineTestCase):
    def test_company_names_become_safe_slugs(self):
        self.assertEqual(slugify("Acme Corp"), "acme-corp")
        self.assertEqual(slugify("  Café & Fils!  "), "cafe-fils")
        self.assertEqual(slugify("***"), "tenant")


class TenantCreateTests(EngineTestCase):
    def test_create_returns_a_key_that_authenticates(self):
        store = get_tenant_store()
        tenant, api_key = store.create("Acme Corp", "anthropic", "sk-ant-key")

        self.assertEqual(tenant.slug, "acme-corp")
        self.assertEqual(tenant.model, DEFAULT_MODELS["anthropic"])
        self.assertEqual(store.authenticate(api_key).slug, "acme-corp")

    def test_gemini_tenant_gets_gemini_default_model(self):
        tenant, _ = get_tenant_store().create("Globex", "gemini", "AIza-key")
        self.assertEqual(tenant.model, DEFAULT_MODELS["gemini"])

    def test_explicit_model_wins_over_default(self):
        tenant, _ = get_tenant_store().create(
            "Initech", "anthropic", "sk", model="claude-sonnet-5"
        )
        self.assertEqual(tenant.model, "claude-sonnet-5")

    def test_duplicate_slug_is_refused(self):
        store = get_tenant_store()
        store.create("Acme", "anthropic", "sk-1")
        with self.assertRaises(TenantExistsError):
            store.create("Acme", "anthropic", "sk-2")

    def test_provider_must_be_known(self):
        with self.assertRaises(ConfigurationError):
            get_tenant_store().create("Acme", "openai", "sk")

    def test_provider_key_is_required(self):
        with self.assertRaises(ConfigurationError):
            get_tenant_store().create("Acme", "anthropic", "  ")

    def test_provider_key_never_lands_in_tenant_json(self):
        store = get_tenant_store()
        tenant, _ = store.create("Acme", "anthropic", "sk-ant-topsecret")
        public = (tenant.home / "tenant.json").read_text()
        self.assertNotIn("sk-ant-topsecret", public)
        self.assertEqual(store.provider_api_key(tenant), "sk-ant-topsecret")


class TenantLookupTests(EngineTestCase):
    def test_unknown_tenant_raises(self):
        with self.assertRaises(TenantNotFoundError):
            get_tenant_store().get("nobody")

    def test_traversal_shaped_slug_raises(self):
        with self.assertRaises(TenantNotFoundError):
            get_tenant_store().get("../../etc")

    def test_list_returns_every_tenant_in_order(self):
        store = get_tenant_store()
        store.create("Beta", "anthropic", "sk-b")
        store.create("Alpha", "gemini", "sk-a")
        self.assertEqual([t.slug for t in store.list()], ["alpha", "beta"])

    def test_config_changes_are_picked_up_without_restart(self):
        store = get_tenant_store()
        tenant, _ = store.create("Acme", "anthropic", "sk")
        # Simulate another worker (a different store instance) updating it.
        other = get_tenant_store()
        other.update("acme", model="claude-sonnet-5")
        self.assertEqual(store.get("acme").model, "claude-sonnet-5")


class TenantAuthTests(EngineTestCase):
    def test_bad_keys_are_refused(self):
        store = get_tenant_store()
        store.create("Acme", "anthropic", "sk")
        for bad in ("", "garbage", "erag.acme.wrongsecret", "erag.other.secret"):
            with self.assertRaises(AuthenticationError):
                store.authenticate(bad)

    def test_one_tenants_key_never_opens_another(self):
        store = get_tenant_store()
        _, acme_key = store.create("Acme", "anthropic", "sk-a")
        store.create("Globex", "gemini", "sk-g")
        # Try to graft acme's secret onto globex's slug.
        secret = acme_key.split(".")[2]
        with self.assertRaises(AuthenticationError):
            store.authenticate(f"erag.globex.{secret}")

    def test_public_dict_reveals_no_secrets(self):
        store = get_tenant_store()
        tenant, api_key = store.create("Acme", "anthropic", "sk")
        public = json.dumps(tenant.public_dict())
        self.assertNotIn(tenant.api_key_hash, public)
        self.assertNotIn(api_key, public)


class TenantUpdateTests(EngineTestCase):
    def test_update_changes_only_allowed_fields(self):
        store = get_tenant_store()
        store.create("Acme", "anthropic", "sk")
        updated = store.update("acme", model="claude-sonnet-5", chat_enabled=False)
        self.assertEqual(updated.model, "claude-sonnet-5")
        self.assertFalse(updated.chat_enabled)
        with self.assertRaises(ConfigurationError):
            store.update("acme", api_key_hash="0" * 64)

    def test_rotate_provider_key(self):
        store = get_tenant_store()
        tenant, _ = store.create("Acme", "anthropic", "sk-old")
        store.rotate_provider_key("acme", "sk-new")
        self.assertEqual(store.provider_api_key(tenant), "sk-new")
