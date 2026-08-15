"""Shared test scaffolding.

Every test runs against a throwaway var/ directory and fake models, so the
suite touches no network, downloads nothing and spends nothing. The fakes are
deliberately simple: the embedder hashes words into buckets (so overlapping
words really are nearby in vector space, which makes retrieval assertions
meaningful) and the provider echoes a canned reply while recording what it was
asked.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Optional

import numpy as np
from django.test import SimpleTestCase

from ragengine.config import get_settings
from ragengine.embeddings import get_embedder
from ragengine.pipeline import clear_pipelines
from ragengine.providers.base import ChatProvider, Completion

ADMIN_KEY = "test-admin-key"

_ENV_KEYS = (
    "ENTERPRISE_VAR_DIR",
    "ENTERPRISE_ADMIN_API_KEY",
    "ENTERPRISE_SECRET_KEY",
    "ENTERPRISE_HISTORY_DB",
    "ENTERPRISE_HISTORY_TURNS",
)


class FakeEmbedder:
    """Deterministic bag-of-words hashing — no model download, real similarity.

    Uses crc32 rather than ``hash()``: Python randomizes string hashes per
    process, which would make retrieval assertions flaky.
    """

    model_name = "fake"
    DIM = 128

    def encode(self, texts: List[str]) -> np.ndarray:
        import zlib

        vectors = np.zeros((len(texts), self.DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,!?;:'\"()")
                if word:
                    vectors[row, zlib.crc32(word.encode()) % self.DIM] += 1.0
        return vectors

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])


class FakeProvider(ChatProvider):
    """Returns canned text and remembers every call for assertions."""

    name = "fake"

    def __init__(self, reply: str = "canned answer", condensed: Optional[str] = None) -> None:
        super().__init__(api_key="unused", model="fake-model", max_tokens=64)
        self.reply = reply
        # What to return when asked to condense; None means echo the reply.
        self.condensed = condensed
        self.calls: List[Dict] = []

    def complete(self, system: str, messages: List[Dict[str, str]]) -> Completion:
        self.calls.append({"system": system, "messages": list(messages)})
        is_condense = "standalone" in system
        text = self.condensed if (is_condense and self.condensed is not None) else self.reply
        return Completion(
            text=text,
            model=self.model,
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )


class EngineTestCase(SimpleTestCase):
    """Points the engine at a temp directory and clears every process cache."""

    def setUp(self) -> None:
        super().setUp()
        self._saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
        self.var_dir = tempfile.mkdtemp(prefix="erag-test-")
        os.environ["ENTERPRISE_VAR_DIR"] = self.var_dir
        os.environ["ENTERPRISE_ADMIN_API_KEY"] = ADMIN_KEY
        os.environ.pop("ENTERPRISE_SECRET_KEY", None)
        os.environ.pop("ENTERPRISE_HISTORY_DB", None)
        os.environ.pop("ENTERPRISE_HISTORY_TURNS", None)
        self._reset_caches()

        # History's in-memory fallback is module state; start each test clean.
        from ragengine import history

        history._MEMORY.clear()

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._reset_caches()
        shutil.rmtree(self.var_dir, ignore_errors=True)
        super().tearDown()

    @staticmethod
    def _reset_caches() -> None:
        get_settings.cache_clear()
        get_embedder.cache_clear()
        clear_pipelines()
