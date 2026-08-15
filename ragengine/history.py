"""Where a conversation lives between requests — keyed by tenant.

The chat APIs are stateless: "memory" only ever means resending the earlier
turns. Messages are stored in the provider-neutral shape the providers accept
(``{"role": "user"|"assistant", "content": str}``), so a tenant can even be
moved between Anthropic and Gemini mid-conversation.

Two backends. In-process memory is the default and is fine for one dev server;
under gunicorn's multiple workers it silently loses half the follow-ups, so
production sets ``ENTERPRISE_HISTORY_DB`` to a SQLite path that every worker
shares. Plain stdlib ``sqlite3`` — no ORM for one table.

Every row carries the tenant slug, and every read filters on it: conversation
ids are minted by browsers, so two tenants can collide on the same UUID
without ever seeing each other's turns.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Settings, get_settings

# Conversation ids are minted by the browser and become storage keys, so they
# are validated rather than trusted. A UUID passes; a path or an essay does not.
CONVERSATION_ID = re.compile(r"\A[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")

Message = Dict[str, str]

# Only used when no DB is configured; see the module docstring.
_MEMORY: Dict[Tuple[str, str], List[Message]] = {}
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT NOT NULL,
    conversation TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_lookup ON messages (tenant, conversation, id);
"""


def is_valid_conversation_id(value: str) -> bool:
    """True for a UUID-shaped id, which is all we ever accept."""
    return bool(value) and bool(CONVERSATION_ID.match(value))


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def recent_messages(
    tenant_slug: str,
    conversation_id: str,
    settings: Optional[Settings] = None,
) -> List[Message]:
    """The last few turns, oldest first.

    Bounding what we *read* is what protects the context window. The store
    itself keeps everything — retention is a policy question, not a prompt one.
    """
    settings = settings or get_settings()
    limit = max(settings.history_turns, 0) * 2
    if not limit:
        return []

    if settings.history_db:
        with _connect(settings.history_db) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages"
                " WHERE tenant = ? AND conversation = ?"
                " ORDER BY id DESC LIMIT ?",
                (tenant_slug, conversation_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    with _LOCK:
        return list(_MEMORY.get((tenant_slug, conversation_id), []))[-limit:]


def record_turn(
    tenant_slug: str,
    conversation_id: str,
    question: str,
    answer: str,
    settings: Optional[Settings] = None,
) -> None:
    """Append one exchange to the store."""
    settings = settings or get_settings()
    turn = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    if settings.history_db:
        with _connect(settings.history_db) as conn:
            conn.executemany(
                "INSERT INTO messages (tenant, conversation, role, content) VALUES (?, ?, ?, ?)",
                [(tenant_slug, conversation_id, m["role"], m["content"]) for m in turn],
            )
        return

    with _LOCK:
        _MEMORY.setdefault((tenant_slug, conversation_id), []).extend(turn)


def forget(tenant_slug: str, conversation_id: str, settings: Optional[Settings] = None) -> None:
    """Drop a conversation entirely."""
    settings = settings or get_settings()
    if settings.history_db:
        with _connect(settings.history_db) as conn:
            conn.execute(
                "DELETE FROM messages WHERE tenant = ? AND conversation = ?",
                (tenant_slug, conversation_id),
            )
        return
    with _LOCK:
        _MEMORY.pop((tenant_slug, conversation_id), None)
