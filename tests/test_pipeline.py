"""The pipeline end to end: index, retrieve, condense, answer, remember.

Runs with the fake embedder patched in and a fake provider, so the tests
exercise real FAISS indexes and real tenant directories with no downloads and
no API spend.
"""

from unittest.mock import patch

from ragengine.exceptions import IndexNotBuiltError, NoDocumentsError
from ragengine.indexing import build_index, save_document
from ragengine.pipeline import TenantPipeline, get_pipeline
from ragengine.store import VectorStore
from ragengine.tenants import get_tenant_store

from .base import EngineTestCase, FakeEmbedder, FakeProvider

CONV = "12345678-1234-1234-1234-123456789abc"

DOCS = {
    "refunds.md": "Refunds are issued within 14 days of purchase.\n\n"
    "Contact support to start a refund request.",
    "shipping.md": "Orders ship worldwide from our warehouse in Rotterdam.\n\n"
    "Express shipping takes two business days.",
}


class PipelineTestCase(EngineTestCase):
    """Builds one tenant with a real (fake-embedded) index."""

    def setUp(self):
        super().setUp()
        self.embedder = FakeEmbedder()
        self._patches = [
            patch("ragengine.store.get_embedder", return_value=self.embedder),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

        self.store = get_tenant_store()
        self.tenant, self.api_key = self.store.create("Acme", "anthropic", "sk-test")
        for filename, text in DOCS.items():
            save_document(self.tenant, filename, text)
        build_index(self.tenant, embedder=self.embedder)

    def pipeline(self, provider=None) -> TenantPipeline:
        return TenantPipeline(
            self.tenant,
            provider or FakeProvider(),
            VectorStore.load(self.tenant.index_path, self.tenant.chunks_path),
        )


class IndexingTests(PipelineTestCase):
    def test_index_covers_every_paragraph(self):
        store = VectorStore.load(self.tenant.index_path, self.tenant.chunks_path)
        self.assertEqual(len(store), 4)

    def test_no_documents_is_an_explicit_error(self):
        tenant, _ = self.store.create("Empty Co", "anthropic", "sk")
        with self.assertRaises(NoDocumentsError):
            build_index(tenant, embedder=self.embedder)

    def test_loading_a_missing_index_is_an_explicit_error(self):
        tenant, _ = self.store.create("Fresh Co", "anthropic", "sk")
        with self.assertRaises(IndexNotBuiltError):
            VectorStore.load(tenant.index_path, tenant.chunks_path)


class RetrievalTests(PipelineTestCase):
    def test_retrieval_finds_the_on_topic_chunk(self):
        results = self.pipeline().retrieve("refund request support", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "refunds.md")

    def test_top_k_defaults_to_tenant_setting(self):
        results = self.pipeline().retrieve("shipping")
        self.assertEqual(len(results), self.tenant.top_k)


class AnswerTests(PipelineTestCase):
    def test_answer_grounds_the_model_in_retrieved_context(self):
        provider = FakeProvider(reply="Refunds take 14 days.")
        answer = self.pipeline(provider).answer("How do refunds work?", k=2)

        self.assertEqual(answer.text, "Refunds take 14 days.")
        self.assertEqual(answer.provider, "fake")
        self.assertEqual(len(provider.calls), 1)  # no history -> no condense call
        self.assertIn("Refunds are issued", provider.calls[0]["system"])
        self.assertIn("Acme", provider.calls[0]["system"])
        self.assertEqual(
            provider.calls[0]["messages"], [{"role": "user", "content": "How do refunds work?"}]
        )

    def test_sources_ride_along_in_the_payload(self):
        payload = self.pipeline().answer("refund request", k=1).to_dict()
        self.assertEqual(payload["sources"][0]["source"], "refunds.md")
        self.assertIn("usage", payload)

    def test_conversation_records_and_replays_turns(self):
        provider = FakeProvider(reply="Within 14 days.", condensed="refund request support")
        pipeline = self.pipeline(provider)

        pipeline.answer("How do refunds work?", conversation_id=CONV)
        pipeline.answer("and how do I start one?", conversation_id=CONV)

        # Second question: one condense call + one answer call, with history.
        self.assertEqual(len(provider.calls), 3)
        final = provider.calls[-1]["messages"]
        self.assertEqual(final[0]["content"], "How do refunds work?")
        self.assertEqual(final[1]["content"], "Within 14 days.")
        self.assertEqual(final[-1]["content"], "and how do I start one?")

    def test_condense_rewrite_drives_retrieval_not_the_answer(self):
        provider = FakeProvider(reply="ok", condensed="refund request support")
        pipeline = self.pipeline(provider)
        pipeline.answer("first question", conversation_id=CONV)
        provider.calls.clear()

        answer = pipeline.answer("and that thing?", k=1, conversation_id=CONV)
        # Retrieval followed the rewrite...
        self.assertEqual(answer.sources[0].source, "refunds.md")
        # ...but the model was asked the user's own words.
        self.assertEqual(provider.calls[-1]["messages"][-1]["content"], "and that thing?")


class PipelineCacheTests(PipelineTestCase):
    def test_pipeline_is_reused_until_the_index_changes(self):
        with patch("ragengine.pipeline.build_provider", return_value=FakeProvider()):
            first = get_pipeline(self.tenant)
            self.assertIs(get_pipeline(self.tenant), first)

            save_document(self.tenant, "returns.md", "Returns are free for 30 days.")
            build_index(self.tenant, embedder=self.embedder)
            rebuilt = get_pipeline(self.tenant)

        self.assertIsNot(rebuilt, first)
        self.assertEqual(len(rebuilt.store), 5)

    def test_config_change_also_rebuilds(self):
        with patch("ragengine.pipeline.build_provider", return_value=FakeProvider()):
            first = get_pipeline(self.tenant)
            updated = self.store.update("acme", model="claude-sonnet-5")
            self.assertIsNot(get_pipeline(updated), first)
