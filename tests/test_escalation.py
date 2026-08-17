"""Support escalation: the file record, the email, and the HTTP boundary.

SMTP is faked at the smtplib seam, so nothing here opens a connection — the
assertions read the EmailMessage the code composed.
"""

import json
import os
from unittest.mock import MagicMock, patch

from ragengine.config import get_settings
from ragengine.escalation import ESCALATIONS_FILENAME, record_escalation, send_escalation_email
from ragengine.exceptions import ConfigurationError
from ragengine.tenants import get_tenant_store

from .base import EngineTestCase

TRANSCRIPT = [
    {"role": "user", "content": "what is the refund window?"},
    {"role": "assistant", "content": "I don't know based on the documents."},
]


def make_tenant(support_email="help@acme.test"):
    tenant, _ = get_tenant_store().create(
        "Acme Corp", "anthropic", "sk-ant-secret", support_email=support_email
    )
    return tenant


class SupportEmailFieldTests(EngineTestCase):
    def test_support_email_round_trips_through_the_record(self):
        make_tenant()
        reloaded = get_tenant_store().get("acme-corp")
        self.assertEqual(reloaded.support_email, "help@acme.test")
        self.assertEqual(reloaded.public_dict()["support_email"], "help@acme.test")

    def test_invalid_support_email_is_refused(self):
        with self.assertRaises(ConfigurationError):
            make_tenant(support_email="not-an-email")

    def test_support_email_is_optional(self):
        tenant = make_tenant(support_email=None)
        self.assertEqual(tenant.support_email, "")


class RecordEscalationTests(EngineTestCase):
    def test_escalation_is_appended_to_the_tenant_log(self):
        tenant = make_tenant()
        record_escalation(tenant, "Priya", "priya@example.com", "refunds?", TRANSCRIPT)
        record_escalation(tenant, "Ravi", "ravi@example.com", "shipping?")
        lines = (tenant.home / ESCALATIONS_FILENAME).read_text().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["name"], "Priya")
        self.assertEqual(first["question"], "refunds?")
        self.assertEqual(first["transcript"], TRANSCRIPT)


class SendEscalationEmailTests(EngineTestCase):
    def _with_smtp(self):
        os.environ["ENTERPRISE_SMTP_HOST"] = "smtp.test"
        os.environ["ENTERPRISE_SMTP_FROM"] = "bot@erag.test"
        get_settings.cache_clear()

    def test_without_smtp_config_nothing_is_sent(self):
        tenant = make_tenant()
        self.assertFalse(send_escalation_email(tenant, "P", "p@example.com", "q"))

    def test_email_carries_contact_question_and_transcript(self):
        self._with_smtp()
        tenant = make_tenant()
        with patch("smtplib.SMTP") as smtp:
            server = MagicMock()
            smtp.return_value.__enter__.return_value = server
            sent = send_escalation_email(
                tenant, "Priya", "priya@example.com", "what is the refund window?", TRANSCRIPT
            )
        self.assertTrue(sent)
        message = server.send_message.call_args[0][0]
        self.assertEqual(message["To"], "help@acme.test")
        self.assertEqual(message["Reply-To"], "priya@example.com")
        self.assertIn("Acme Corp", message["Subject"])
        body = message.get_content()
        self.assertIn("what is the refund window?", body)
        self.assertIn("Visitor: what is the refund window?", body)


class EscalateEndpointTests(EngineTestCase):
    def setUp(self):
        super().setUp()
        self.tenant = make_tenant()

    def escalate(self, payload, slug="acme-corp"):
        return self.client.post(
            f"/chat/{slug}/escalate/", json.dumps(payload), content_type="application/json"
        )

    def valid_payload(self, **overrides):
        payload = {"name": "Priya", "email": "priya@example.com", "question": "refunds?"}
        payload.update(overrides)
        return payload

    def test_escalation_is_recorded_without_smtp(self):
        response = self.escalate(self.valid_payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "recorded"})
        self.assertTrue((self.tenant.home / ESCALATIONS_FILENAME).is_file())

    def test_validation_is_enforced_at_the_boundary(self):
        cases = [
            self.valid_payload(name=""),
            self.valid_payload(name="x" * 81),
            self.valid_payload(email="nope"),
            self.valid_payload(question=""),
            self.valid_payload(question="x" * 2001),
        ]
        for payload in cases:
            self.assertEqual(self.escalate(payload).status_code, 400, payload)

    def test_unknown_tenant_and_missing_support_email(self):
        self.assertEqual(self.escalate(self.valid_payload(), slug="nobody").status_code, 404)
        get_tenant_store().update("acme-corp", support_email="")
        self.assertEqual(self.escalate(self.valid_payload()).status_code, 400)

    def test_mail_failure_still_records_and_returns_200(self):
        os.environ["ENTERPRISE_SMTP_HOST"] = "smtp.test"
        os.environ["ENTERPRISE_SMTP_FROM"] = "bot@erag.test"
        get_settings.cache_clear()
        with patch("smtplib.SMTP", side_effect=OSError("mail server down")):
            response = self.escalate(self.valid_payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "recorded"})
        self.assertTrue((self.tenant.home / ESCALATIONS_FILENAME).is_file())


class ChatPageSupportTests(EngineTestCase):
    def test_support_ui_only_renders_when_configured(self):
        make_tenant()
        with_support = self.client.get("/chat/acme-corp/")
        self.assertContains(with_support, "Email support")

        get_tenant_store().create("No Help Co", "anthropic", "k")
        without = self.client.get("/chat/no-help-co/")
        self.assertNotContains(without, "Email support")
