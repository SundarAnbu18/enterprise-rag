"""The operator console: its admin endpoints and the page shell.

Same philosophy as test_views — the embedder is faked at the store seam, so
uploads really chunk and index, but nothing downloads a model or leaves the
process.
"""

import json
from unittest.mock import patch

from ragengine.tenants import get_tenant_store

from .base import ADMIN_KEY, EngineTestCase, FakeEmbedder

DOC = "Refunds take 14 days.\n\nShipping is free."


class ConsoleApiTestCase(EngineTestCase):
    def setUp(self):
        super().setUp()
        self.tenant, self.api_key = get_tenant_store().create(
            "Acme Corp", "anthropic", "sk-ant-secret"
        )

    def admin_get(self, url, key=ADMIN_KEY):
        return self.client.get(url, HTTP_X_ADMIN_KEY=key)

    def admin_post(self, url, payload, key=ADMIN_KEY):
        return self.client.post(
            url, json.dumps(payload), content_type="application/json", HTTP_X_ADMIN_KEY=key
        )

    def upload(self, filename="faq.md", text=DOC):
        with patch("ragengine.store.get_embedder", return_value=FakeEmbedder()):
            return self.admin_post(
                f"/api/v1/tenants/{self.tenant.slug}/documents/",
                {"filename": filename, "text": text},
            )


class TenantListVectorsTests(ConsoleApiTestCase):
    def test_list_reports_backend_and_vector_counts(self):
        self.upload()
        response = self.admin_get("/api/v1/tenants/")
        body = response.json()
        self.assertEqual(body["backend"], "faiss")
        self.assertEqual(body["tenants"][0]["vectors"], 2)
        self.assertEqual(body["tenants"][0]["chat_url"], "/chat/acme-corp/")
        self.assertTrue(body["tenants"][0]["index_ready"])


class TenantDetailTests(ConsoleApiTestCase):
    def test_detail_lists_documents_with_chunk_counts(self):
        self.upload()
        response = self.admin_get("/api/v1/tenants/acme-corp/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["vectors"], 2)
        self.assertEqual(body["backend"], "faiss")
        self.assertEqual(len(body["documents"]), 1)
        doc = body["documents"][0]
        self.assertEqual(doc["name"], "faq.md")
        self.assertEqual(doc["chunks"], 2)
        self.assertGreater(doc["bytes"], 0)

    def test_detail_of_unknown_tenant_is_404(self):
        self.assertEqual(self.admin_get("/api/v1/tenants/nobody/").status_code, 404)

    def test_detail_requires_the_admin_key(self):
        self.assertEqual(
            self.admin_get("/api/v1/tenants/acme-corp/", key="wrong").status_code, 401
        )


class AdminUploadTests(ConsoleApiTestCase):
    def test_admin_can_upload_and_index_for_any_tenant(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"status": "indexed", "chunks": 2})
        self.assertTrue(self.tenant.index_path.is_file())

    def test_upload_to_unknown_tenant_is_404(self):
        response = self.admin_post(
            "/api/v1/tenants/nobody/documents/", {"filename": "a.md", "text": "x"}
        )
        self.assertEqual(response.status_code, 404)

    def test_hostile_filenames_are_refused(self):
        response = self.admin_post(
            "/api/v1/tenants/acme-corp/documents/",
            {"filename": "../evil.md", "text": "hi"},
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_requires_the_admin_key(self):
        response = self.admin_post(
            "/api/v1/tenants/acme-corp/documents/",
            {"filename": "a.md", "text": "x"},
            key="wrong",
        )
        self.assertEqual(response.status_code, 401)


class ChatEnabledAtCreationTests(EngineTestCase):
    def test_console_can_disable_chat_at_onboarding(self):
        response = self.client.post(
            "/api/v1/tenants/",
            json.dumps(
                {
                    "name": "Quiet Co",
                    "provider": "anthropic",
                    "provider_api_key": "sk-ant-secret",
                    "chat_enabled": False,
                }
            ),
            content_type="application/json",
            HTTP_X_ADMIN_KEY=ADMIN_KEY,
        )
        self.assertEqual(response.status_code, 201)
        self.assertFalse(response.json()["chat_enabled"])
        self.assertEqual(self.client.get("/chat/quiet-co/").status_code, 403)


class ConsolePageTests(EngineTestCase):
    def test_console_shell_is_served_without_auth(self):
        response = self.client.get("/console/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "RAG Console")

    def test_console_shell_contains_no_secrets(self):
        response = self.client.get("/console/")
        self.assertNotContains(response, ADMIN_KEY)
