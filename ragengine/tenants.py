"""The tenant registry: who the customers are and where their data lives.

A tenant is a directory, not a database row. ``var/tenants/<slug>/`` holds:

    tenant.json    public configuration (name, provider, model, key hash)
    secrets.json   the tenant's provider API key, encrypted (see crypto.py)
    documents/     the corpus they upload
    index/         the FAISS index built from it

Files were chosen over a database deliberately: tenant records change only at
onboarding time, reads are cheap, every gunicorn worker sees the same disk, and
there is no schema to migrate. Reads go through an mtime-based cache so a
tenant updated by one worker (or the CLI) is picked up by the others without a
restart. If the fleet ever outgrows one machine, this module is the only thing
that needs a database behind it — its callers only see ``Tenant`` objects.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Settings, get_settings
from .crypto import (
    decrypt_secret,
    encrypt_secret,
    generate_tenant_key,
    hash_key,
    slug_from_key,
    verify_key,
)
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    TenantExistsError,
    TenantNotFoundError,
)

PROVIDERS = ("anthropic", "gemini")

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "gemini": "gemini-2.5-flash",
}

# Slugs become directory names and URL segments, so they are strictly shaped.
SLUG_RE = re.compile(r"\A[a-z0-9][a-z0-9-]{0,62}\Z")

# Deliberately loose: enough to catch typos, not to adjudicate RFC 5322.
EMAIL_RE = re.compile(r"\A[^@\s]+@[^@\s]+\.[^@\s]+\Z")


def slugify(name: str) -> str:
    """Turn a company name into a filesystem- and URL-safe slug."""
    value = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:63] or "tenant"


@dataclass(frozen=True)
class Tenant:
    """One customer's resolved configuration."""

    slug: str
    name: str
    provider: str
    model: str
    api_key_hash: str
    top_k: int
    max_tokens: int
    chat_enabled: bool = True
    # Where unanswerable questions escalate to. Empty means the tenant has
    # no human fallback and the chat page offers none.
    support_email: str = ""
    created_at: str = ""
    home: Path = field(default=Path("."), compare=False)

    @property
    def documents_dir(self) -> Path:
        return self.home / "documents"

    @property
    def index_dir(self) -> Path:
        return self.home / "index"

    @property
    def index_path(self) -> Path:
        """The FAISS index itself."""
        return self.index_dir / "index.faiss"

    @property
    def chunks_path(self) -> Path:
        """The chunk texts, in the same order as the vectors in the index."""
        return self.index_dir / "chunks.json"

    def to_dict(self) -> dict:
        """The JSON written to tenant.json — everything except secrets."""
        return {
            "slug": self.slug,
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "api_key_hash": self.api_key_hash,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "chat_enabled": self.chat_enabled,
            "support_email": self.support_email,
            "created_at": self.created_at,
        }

    def public_dict(self) -> dict:
        """What API responses may reveal — never the key hash."""
        # Imported here so the registry stays importable on its own; vectordb
        # refers back to Tenant in its type hints.
        from .vectordb import index_stamp_path

        return {
            "slug": self.slug,
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "chat_enabled": self.chat_enabled,
            "support_email": self.support_email,
            "created_at": self.created_at,
            "index_ready": index_stamp_path(self).is_file(),
        }


class TenantStore:
    """Reads and writes the tenant directory tree."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        # slug -> (tenant.json mtime, Tenant); consulted before touching disk.
        self._cache: Dict[str, Tuple[float, Tenant]] = {}

    # -- reading ------------------------------------------------------------

    def _tenant_file(self, slug: str) -> Path:
        return self.settings.tenants_dir / slug / "tenant.json"

    def get(self, slug: str) -> Tenant:
        """Load one tenant, from cache when the file hasn't changed."""
        if not SLUG_RE.match(slug or ""):
            raise TenantNotFoundError(f"No tenant {slug!r}")

        path = self._tenant_file(slug)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self._cache.pop(slug, None)
            raise TenantNotFoundError(f"No tenant {slug!r}")

        cached = self._cache.get(slug)
        if cached and cached[0] == mtime:
            return cached[1]

        data = json.loads(path.read_text(encoding="utf-8"))
        tenant = Tenant(
            slug=data["slug"],
            name=data.get("name", data["slug"]),
            provider=data["provider"],
            model=data["model"],
            api_key_hash=data.get("api_key_hash", ""),
            top_k=int(data.get("top_k", self.settings.default_top_k)),
            max_tokens=int(data.get("max_tokens", self.settings.default_max_tokens)),
            chat_enabled=bool(data.get("chat_enabled", True)),
            support_email=data.get("support_email", ""),
            created_at=data.get("created_at", ""),
            home=path.parent,
        )
        self._cache[slug] = (mtime, tenant)
        return tenant

    def list(self) -> List[Tenant]:
        """Every tenant, in slug order."""
        root = self.settings.tenants_dir
        if not root.is_dir():
            return []
        tenants = []
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and (entry / "tenant.json").is_file():
                tenants.append(self.get(entry.name))
        return tenants

    def authenticate(self, api_key: str) -> Tenant:
        """Resolve a presented API key to its tenant, or refuse.

        The slug embedded in the key makes this a single directory lookup; the
        digest comparison is constant-time, so neither a wrong slug nor a wrong
        secret leaks anything through timing.
        """
        slug = slug_from_key(api_key or "")
        if slug is None:
            raise AuthenticationError("Invalid API key")
        try:
            tenant = self.get(slug)
        except TenantNotFoundError:
            raise AuthenticationError("Invalid API key")
        if not tenant.api_key_hash or not verify_key(api_key, tenant.api_key_hash):
            raise AuthenticationError("Invalid API key")
        return tenant

    def provider_api_key(self, tenant: Tenant) -> str:
        """The tenant's own LLM credential, decrypted for use."""
        path = tenant.home / "secrets.json"
        if not path.is_file():
            raise ConfigurationError(f"Tenant {tenant.slug!r} has no provider key on file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return decrypt_secret(payload["provider_api_key"])

    # -- writing ------------------------------------------------------------

    def create(
        self,
        name: str,
        provider: str,
        provider_api_key: str,
        model: Optional[str] = None,
        slug: Optional[str] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        chat_enabled: bool = True,
        support_email: Optional[str] = None,
    ) -> Tuple[Tenant, str]:
        """Onboard a customer. Returns the tenant and their API key —
        the only time the key ever exists in plaintext on our side."""
        provider = (provider or "").strip().lower()
        if provider not in PROVIDERS:
            raise ConfigurationError(
                f"provider must be one of {', '.join(PROVIDERS)}; got {provider!r}"
            )
        if not (provider_api_key or "").strip():
            raise ConfigurationError("provider_api_key is required")
        if not (name or "").strip():
            raise ConfigurationError("name is required")
        support_email = (support_email or "").strip()
        if support_email and not EMAIL_RE.match(support_email):
            raise ConfigurationError(f"invalid support_email {support_email!r}")

        slug = (slug or slugify(name)).strip().lower()
        if not SLUG_RE.match(slug):
            raise ConfigurationError(f"invalid slug {slug!r}")

        home = self.settings.tenants_dir / slug
        if (home / "tenant.json").exists():
            raise TenantExistsError(f"Tenant {slug!r} already exists")

        api_key = generate_tenant_key(slug)
        tenant = Tenant(
            slug=slug,
            name=name.strip(),
            provider=provider,
            model=(model or "").strip() or DEFAULT_MODELS[provider],
            api_key_hash=hash_key(api_key),
            top_k=top_k or self.settings.default_top_k,
            max_tokens=max_tokens or self.settings.default_max_tokens,
            chat_enabled=chat_enabled,
            support_email=support_email,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            home=home,
        )

        (home / "documents").mkdir(parents=True, exist_ok=True)
        (home / "index").mkdir(parents=True, exist_ok=True)
        self._write_json(home / "tenant.json", tenant.to_dict())
        self._write_json(
            home / "secrets.json",
            {"provider_api_key": encrypt_secret(provider_api_key.strip())},
            private=True,
        )
        return tenant, api_key

    def update(self, slug: str, **changes) -> Tenant:
        """Change a tenant's public configuration in place."""
        tenant = self.get(slug)
        allowed = {
            "name", "model", "top_k", "max_tokens", "chat_enabled", "provider", "support_email"
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ConfigurationError(f"cannot update field(s): {', '.join(sorted(unknown))}")
        if "provider" in changes and changes["provider"] not in PROVIDERS:
            raise ConfigurationError(f"provider must be one of {', '.join(PROVIDERS)}")
        if changes.get("support_email") and not EMAIL_RE.match(changes["support_email"]):
            raise ConfigurationError(f"invalid support_email {changes['support_email']!r}")
        updated = replace(tenant, **changes)
        self._write_json(self._tenant_file(slug), updated.to_dict())
        self._cache.pop(slug, None)
        return self.get(slug)

    def rotate_provider_key(self, slug: str, provider_api_key: str) -> None:
        """Replace the tenant's stored LLM credential."""
        tenant = self.get(slug)
        if not (provider_api_key or "").strip():
            raise ConfigurationError("provider_api_key is required")
        self._write_json(
            tenant.home / "secrets.json",
            {"provider_api_key": encrypt_secret(provider_api_key.strip())},
            private=True,
        )

    @staticmethod
    def _write_json(path: Path, payload: dict, private: bool = False) -> None:
        """Write atomically so a crashed request never leaves a half-file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if private:
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)


def get_tenant_store() -> TenantStore:
    """A store bound to the current settings.

    Built per call rather than cached: construction is a couple of attribute
    assignments, and it means tests that swap settings never see a stale store.
    The per-tenant read cache lives inside pipeline-level callers that keep one
    store around.
    """
    return TenantStore()
