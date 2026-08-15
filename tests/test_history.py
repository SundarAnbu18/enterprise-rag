"""Conversation history: both backends, bounded reads, tenant isolation."""

import os
from pathlib import Path

from ragengine.config import get_settings
from ragengine.history import forget, is_valid_conversation_id, recent_messages, record_turn

from .base import EngineTestCase

CONV = "12345678-1234-1234-1234-123456789abc"
OTHER = "87654321-4321-4321-4321-cba987654321"


class ConversationIdTests(EngineTestCase):
    def test_uuid_shapes_pass_everything_else_fails(self):
        self.assertTrue(is_valid_conversation_id(CONV))
        for bad in ("", "not-a-uuid", "../../etc/passwd", CONV + "x"):
            self.assertFalse(is_valid_conversation_id(bad), bad)


class MemoryBackendTests(EngineTestCase):
    def test_turns_round_trip(self):
        record_turn("acme", CONV, "hello?", "hi!")
        self.assertEqual(
            recent_messages("acme", CONV),
            [
                {"role": "user", "content": "hello?"},
                {"role": "assistant", "content": "hi!"},
            ],
        )

    def test_tenants_never_see_each_others_turns(self):
        record_turn("acme", CONV, "acme question", "acme answer")
        self.assertEqual(recent_messages("globex", CONV), [])

    def test_reads_are_bounded_by_history_turns(self):
        os.environ["ENTERPRISE_HISTORY_TURNS"] = "2"
        get_settings.cache_clear()
        for i in range(5):
            record_turn("acme", CONV, f"q{i}", f"a{i}")
        messages = recent_messages("acme", CONV)
        self.assertEqual(len(messages), 4)  # 2 turns * 2 messages
        self.assertEqual(messages[0]["content"], "q3")

    def test_forget_drops_the_conversation(self):
        record_turn("acme", CONV, "q", "a")
        forget("acme", CONV)
        self.assertEqual(recent_messages("acme", CONV), [])


class SqliteBackendTests(EngineTestCase):
    def setUp(self):
        super().setUp()
        os.environ["ENTERPRISE_HISTORY_DB"] = str(Path(self.var_dir) / "history.sqlite3")
        get_settings.cache_clear()

    def test_turns_round_trip_through_sqlite(self):
        record_turn("acme", CONV, "hello?", "hi!")
        record_turn("acme", OTHER, "elsewhere", "yes")
        self.assertEqual(
            recent_messages("acme", CONV),
            [
                {"role": "user", "content": "hello?"},
                {"role": "assistant", "content": "hi!"},
            ],
        )

    def test_sqlite_isolates_tenants_too(self):
        record_turn("acme", CONV, "q", "a")
        self.assertEqual(recent_messages("globex", CONV), [])

    def test_forget_deletes_rows(self):
        record_turn("acme", CONV, "q", "a")
        forget("acme", CONV)
        self.assertEqual(recent_messages("acme", CONV), [])
