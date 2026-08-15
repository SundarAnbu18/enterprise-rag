"""The HTTP layer: auth boundaries, status-code mapping, the chat page.

``answer_question`` is mocked at the view seam and the embedder at the store
seam, so these tests cover exactly what the views own — validation, auth and
translation — without models or money.
"""

import json
from unittest.mock import patch

from ragengine.pipeline import Answer
from ragengine.tenants import get_tenant_store

from .base import ADMIN_KEY, EngineTestCase, FakeEmbedder

CONV = "12345678-1234-1234-1234-123456789abc"


def fake_answer(text="the answer"):
    return Answer(
        text=text,
        provider="anthropic",
        model="claude-opus-5",
        input_tokens=10,
        output_tokens=5,
        stop_reason="end_turn",
        sources=[],
    )


class TenantManagementApiTests(EngineTestCase):
    def _create(self, body=None, key=ADMIN_KEY):
        payload = {
            "name": "Acme Corp",
            "provider": "anthropic",
            "provider_api_key": "sk-ant-secret",
        }
        payload.update(body or {})
        headers = {"HTTP_X_ADMIN_KEY": key} if key else {}
        return self.client.post(
            "/api/v1/tenants/", json.dumps(payload), content_type="application/json", **headers
        )

    def test_admin_can_onboard_a_tenant(self):
        response = self._create()
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["slug"], "acme-corp")
        self.assertTrue(body["api_key"].startswith("erag.acme-corp."))
        self.assertEqual(body["chat_url"], "/chat/acme-corp/")
        self.assertNotIn("provider_api_key", body)

    def test_wrong_or_missing_admin_key_is_refused(self):
        self.assertEqual(self._create(key="wrong").status_code, 401)
        self.assertEqual(self._create(key=None).status_code, 401)

    def test_admin_endpoint_fails_closed_when_unconfigured(self):
        import os

        os.environ["ENTERPRISE_ADMIN_API_KEY"] = ""
        from ragengine.config import get_settings

        get_settings.cache_clear()
        self.assertEqual(self._create().status_code, 503)

    def test_validation_errors_are_400(self):
        self.assertEqual(self._create({"provider": "openai"}).status_code, 400)
        self.assertEqual(self._create({"provider_api_key": ""}).status_code, 400)

    def test_duplicate_tenant_is_409(self):
        self._create()
        self.assertEqual(self._create().status_code, 409)

    def test_admin_can_list_tenants(self):
        self._create()
        response = self.client.get("/api/v1/tenants/", HTTP_X_ADMIN_KEY=ADMIN_KEY)
        self.assertEqual(response.status_code, 200)
        tenants = response.json()["tenants"]
        self.assertEqual([t["slug"] for t in tenants], ["acme-corp"])
        self.assertNotIn("api_key_hash", tenants[0])


class TenantApiTestCase(EngineTestCase):
    """Creates one tenant and keeps its key handy."""

    def setUp(self):
        super().setUp()
        self.tenant, self.api_key = get_tenant_store().create(
            "Acme Corp", "anthropic", "sk-ant-secret"
        )

    def post_json(self, url, payload, key=None):
        headers = {"HTTP_X_API_KEY": key or self.api_key}
        return self.client.post(
            url, json.dumps(payload), content_type="application/json", **headers
        )


class DocumentApiTests(TenantApiTestCase):
    def test_upload_stores_and_indexes(self):
        with patch("ragengine.store.get_embedder", return_value=FakeEmbedder()):
            response = self.post_json(
                "/api/v1/documents/",
                {"filename": "faq.md", "text": "Refunds take 14 days.\n\nShipping is free."},
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"status": "indexed", "chunks": 2})
        self.assertTrue(self.tenant.index_path.is_file())

    def test_upload_requires_a_tenant_key(self):
        response = self.post_json("/api/v1/documents/", {"filename": "a.md", "text": "x"}, key="bad")
        self.assertEqual(response.status_code, 401)

    def test_hostile_filenames_are_refused(self):
        for name in ("../evil.md", "/etc/passwd", "no-extension", ".hidden.md"):
            response = self.post_json("/api/v1/documents/", {"filename": name, "text": "hi"})
            self.assertEqual(response.status_code, 400, name)

    def test_reindex_without_documents_is_400(self):
        response = self.post_json("/api/v1/reindex/", {})
        self.assertEqual(response.status_code, 400)


class AskApiTests(TenantApiTestCase):
    def test_returns_the_answer(self):
        with patch("ragapi.views.answer_question", return_value=fake_answer("42")) as mocked:
            response = self.post_json("/api/v1/ask/", {"question": "meaning of life?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "42")
        tenant_arg = mocked.call_args[0][0]
        self.assertEqual(tenant_arg.slug, "acme-corp")

    def test_conversation_id_is_validated_and_echoed(self):
        with patch("ragapi.views.answer_question", return_value=fake_answer()):
            good = self.post_json("/api/v1/ask/", {"question": "q", "conversation_id": CONV})
            bad = self.post_json("/api/v1/ask/", {"question": "q", "conversation_id": "nope"})
        self.assertEqual(good.json()["conversation_id"], CONV)
        self.assertEqual(bad.status_code, 400)

    def test_question_is_required_and_bounded(self):
        self.assertEqual(self.post_json("/api/v1/ask/", {}).status_code, 400)
        self.assertEqual(
            self.post_json("/api/v1/ask/", {"question": "x" * 2001}).status_code, 400
        )

    def test_missing_index_maps_to_409(self):
        from ragengine.exceptions import IndexNotBuiltError

        with patch("ragapi.views.answer_question", side_effect=IndexNotBuiltError("no index")):
            response = self.post_json("/api/v1/ask/", {"question": "q"})
        self.assertEqual(response.status_code, 409)

    def test_provider_failure_maps_to_502(self):
        from ragengine.exceptions import ProviderError

        with patch("ragapi.views.answer_question", side_effect=ProviderError("bad key")):
            response = self.post_json("/api/v1/ask/", {"question": "q"})
        self.assertEqual(response.status_code, 502)

    def test_wrong_key_is_401(self):
        response = self.post_json("/api/v1/ask/", {"question": "q"}, key="erag.acme-corp.wrong")
        self.assertEqual(response.status_code, 401)


class ChatPageTests(TenantApiTestCase):
    def test_page_renders_with_tenant_branding(self):
        response = self.client.get("/chat/acme-corp/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Acme Corp")

    def test_unknown_workspace_is_404(self):
        self.assertEqual(self.client.get("/chat/nobody/").status_code, 404)

    def test_disabled_chat_is_403(self):
        get_tenant_store().update("acme-corp", chat_enabled=False)
        self.assertEqual(self.client.get("/chat/acme-corp/").status_code, 403)

    def test_page_post_answers_without_a_key(self):
        with patch("ragapi.views.answer_question", return_value=fake_answer("hi there")):
            response = self.client.post(
                "/chat/acme-corp/",
                json.dumps({"question": "hello", "conversation_id": CONV}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "hi there")


class HealthTests(EngineTestCase):
    def test_health_is_cheap_and_public(self):
        response = self.client.get("/api/v1/health/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
