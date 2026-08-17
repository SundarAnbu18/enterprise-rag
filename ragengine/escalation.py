"""When the assistant can't help, hand the visitor to a human.

The grounding prompt makes the model admit when the answer isn't in the
tenant's documents — that admission is a dead end for the visitor unless it
goes somewhere. This module is the somewhere: the visitor leaves their name
and email, and the tenant's support team gets the whole exchange.

Two layers, deliberately: every escalation is first appended to
``escalations.jsonl`` in the tenant's directory (same file-based ethos as the
rest of their data — greppable, backed up with the tenant, no queue to run),
and then email delivery is attempted on top when the deployment has SMTP
configured *and* the tenant has a ``support_email``. Recording never depends
on the mail server being up; the caller decides how loudly to report a failed
send.

``smtplib`` is stdlib, so the engine stays dependency- and Django-free.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Sequence

from .config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .tenants import Tenant

ESCALATIONS_FILENAME = "escalations.jsonl"


def _transcript_text(transcript: Sequence[dict]) -> str:
    speaker = {"user": "Visitor", "assistant": "Assistant"}
    return "\n".join(
        f"{speaker.get(m.get('role'), m.get('role'))}: {m.get('content', '')}" for m in transcript
    )


def record_escalation(
    tenant: "Tenant",
    name: str,
    email: str,
    question: str,
    transcript: Sequence[dict] = (),
) -> dict:
    """Append one escalation to the tenant's log. Never sends anything."""
    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "email": email,
        "question": question,
        "transcript": list(transcript),
    }
    path = tenant.home / ESCALATIONS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)  # visitor contact details — same discipline as secrets
    return entry


def can_email(tenant: "Tenant", settings: Optional[Settings] = None) -> bool:
    """Whether an escalation for this tenant would actually reach a mailbox."""
    settings = settings or get_settings()
    return bool(tenant.support_email and settings.smtp_host and settings.smtp_from)


def send_escalation_email(
    tenant: "Tenant",
    name: str,
    email: str,
    question: str,
    transcript: Sequence[dict] = (),
    settings: Optional[Settings] = None,
) -> bool:
    """Mail the escalation to the tenant's support address.

    Returns False when mail is not configured for this tenant; raises the
    underlying ``smtplib``/socket error when sending was attempted and failed,
    so the caller can log it without this module needing a logging policy.
    """
    settings = settings or get_settings()
    if not can_email(tenant, settings):
        return False

    import smtplib
    from email.message import EmailMessage

    lines = [
        f"A visitor on the {tenant.name} assistant needs a human answer.",
        "",
        f"Name:  {name}",
        f"Email: {email}",
        "",
        "Question the assistant could not answer:",
        question,
    ]
    if transcript:
        lines += ["", "Conversation so far:", _transcript_text(transcript)]
    lines += ["", "—", f"Sent by the {tenant.name} RAG assistant. Reply to reach the visitor."]

    message = EmailMessage()
    message["Subject"] = f"[{tenant.name}] Support request from {name}"
    message["From"] = settings.smtp_from
    message["To"] = tenant.support_email
    # Replying goes straight to the visitor, not back to the robot.
    message["Reply-To"] = email
    message.set_content("\n".join(lines))

    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=20) as server:
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as server:
            server.starttls()
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
    return True
