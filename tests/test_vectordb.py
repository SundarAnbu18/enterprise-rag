"""The pluggable vector backend, exercised against Qdrant's in-memory mode.

No server, no network: ``ENTERPRISE_QDRANT_URL=:memory:`` runs the whole
database inside the test process, and the cached client is what makes a
build visible to the search that follows. The Qdrant classes are skipped
wholesale when qdrant-client is not installed — the FAISS default has its
own coverage in test_pipeline and below.
"""

from __future__ import annotations

import json
import os
import unittest

from ragengine.config import get_settings
from ragengine.exceptions import IndexNotBuiltError, VectorDBError
from ragengine.indexing import build_index, save_document
from ragengine.tenants import get_tenant_store
from ragengine.vectordb import (
    MARKER_FILENAME,
    get_qdrant_client,
    index_info,
    index_stamp_path,
    live_vector_counts,
    load_store,
)

from .base import EngineTestCase, FakeEmbedder

try:  # the suite must run on machines that never selected the backend
    import qdrant_client  # noqa: F401

    HAVE_QDRANT = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_QDRANT = False

DOC = "Refunds are processed within 14 days.\n\nShipping is free over fifty dollars."


class FaissDefaultTests(EngineTestCase):
    """The default backend still behaves exactly as before the split."""

    def test_stamp_path_is_the_faiss_index(self):
        tenant, _ = get_tenant_store().create("Acme", "anthropic", "k")
        self.assertEqual(index_stamp_path(tenant), tenant.index_path)

    def test_index_info_counts_chunks_per_source(self):
        tenant, _ = get_tenant_store().create("Acme", "anthropic", "k")
        self.assertIsNone(index_info(tenant))
        save_document(tenant, "faq.md", DOC)
        build_index(tenant, embedder=FakeEmbedder())
        info = index_info(tenant)
        self.assertEqual(info["vectors"], 2)
        self.assertEqual(info["sources"], {"faq.md": 2})

    def test_unknown_backend_is_refused(self):
        os.environ["ENTERPRISE_VECTOR_BACKEND"] = "pinecone"
        get_settings.cache_clear()
        from ragengine.exceptions import ConfigurationError

        with self.assertRaises(ConfigurationError):
            get_settings()


@unittest.skipUnless(HAVE_QDRANT, "qdrant-client is not installed")
class QdrantTestCase(EngineTestCase):
    """Everything here runs against an in-process Qdrant."""

    def setUp(self):
        super().setUp()
        os.environ["ENTERPRISE_VECTOR_BACKEND"] = "qdrant"
        os.environ["ENTERPRISE_QDRANT_URL"] = ":memory:"
        get_settings.cache_clear()
        get_qdrant_client.cache_clear()
        self.tenant, _ = get_tenant_store().create("Acme", "anthropic", "k")

    def _build(self):
        save_document(self.tenant, "faq.md", DOC)
        return build_index(self.tenant, embedder=FakeEmbedder())


class QdrantBackendTests(QdrantTestCase):
    def test_build_then_search_retrieves_the_right_chunk(self):
        store = self._build()
        self.assertEqual(len(store), 2)
        results = load_store(self.tenant).search(
            "how fast are refunds processed", 1, embedder=FakeEmbedder()
        )
        self.assertEqual(len(results), 1)
        self.assertIn("Refunds", results[0].text)
        self.assertEqual(results[0].source, "faq.md")

    def test_stamp_records_counts_and_marks_ready(self):
        self.assertFalse(self.tenant.public_dict()["index_ready"])
        self._build()
        self.assertTrue(self.tenant.public_dict()["index_ready"])
        stamp = json.loads((self.tenant.index_dir / MARKER_FILENAME).read_text())
        self.assertEqual(stamp["vectors"], 2)
        self.assertEqual(stamp["sources"], {"faq.md": 2})
        self.assertEqual(index_info(self.tenant), {"vectors": 2, "sources": {"faq.md": 2}})

    def test_load_before_build_is_index_not_built(self):
        with self.assertRaises(IndexNotBuiltError):
            load_store(self.tenant)

    def test_drifted_collection_is_refused(self):
        self._build()
        stamp_path = self.tenant.index_dir / MARKER_FILENAME
        stamp = json.loads(stamp_path.read_text())
        stamp["vectors"] = 99  # collection and stamp now disagree
        stamp_path.write_text(json.dumps(stamp))
        with self.assertRaises(IndexNotBuiltError):
            load_store(self.tenant)

    def test_rebuild_replaces_rather_than_appends(self):
        self._build()
        self._build()
        self.assertEqual(len(load_store(self.tenant)), 2)

    def test_live_counts_track_the_cluster_not_the_stamps(self):
        self.assertEqual(live_vector_counts(), {})
        self._build()
        self.assertEqual(live_vector_counts(), {"acme": 2})

    def test_listing_endpoint_flags_missing_and_present_collections(self):
        from .base import ADMIN_KEY

        self._build()
        # A second tenant that exists locally but was never indexed here.
        get_tenant_store().create("Ghost Co", "anthropic", "k")
        response = self.client.get("/api/v1/tenants/", HTTP_X_ADMIN_KEY=ADMIN_KEY)
        body = response.json()
        self.assertEqual(body["backend_status"], "connected")
        by_slug = {t["slug"]: t for t in body["tenants"]}
        self.assertTrue(by_slug["acme"]["in_backend"])
        self.assertEqual(by_slug["acme"]["live_vectors"], 2)
        self.assertFalse(by_slug["ghost-co"]["in_backend"])
        self.assertEqual(by_slug["ghost-co"]["live_vectors"], 0)

    def test_listing_survives_an_unreachable_cluster(self):
        from unittest.mock import patch

        from ragengine.exceptions import VectorDBError

        from .base import ADMIN_KEY

        self._build()
        with patch(
            "ragapi.views.live_vector_counts", side_effect=VectorDBError("down")
        ):
            response = self.client.get("/api/v1/tenants/", HTTP_X_ADMIN_KEY=ADMIN_KEY)
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["backend_status"], "unreachable")
        self.assertEqual(len(body["tenants"]), 1)
        self.assertNotIn("in_backend", body["tenants"][0])

    def test_missing_package_reads_as_instructions_not_traceback(self):
        # Simulated by asking for a client while hiding the import.
        import sys
        from unittest.mock import patch

        get_qdrant_client.cache_clear()
        with patch.dict(sys.modules, {"qdrant_client": None}):
            with self.assertRaises(VectorDBError) as caught:
                get_qdrant_client()
        self.assertIn("pip install qdrant-client", str(caught.exception))
        get_qdrant_client.cache_clear()
