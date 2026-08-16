"""Every deployment-wide knob in one place, read from the environment.

The rest of the package never touches ``os.environ`` directly — it asks for
``get_settings()``. Everything *per-tenant* (provider, model, API key, corpus)
lives in the tenant record instead; this file only holds what is shared by the
whole deployment: where tenant data lives, which embedding model to load, and
the operator-level secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .exceptions import ConfigurationError

# enterprise_rag/ragengine/config.py -> enterprise_rag/
PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR.parent


def load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from a .env file into ``os.environ``.

    Values already present in the environment win, so systemd or the shell can
    always override the file. Written by hand to avoid a python-dotenv
    dependency for twelve lines of parsing.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_dotenv() -> None:
    """Pick up enterprise_rag/.env, falling back to the .env at the repo root."""
    load_env_file(PROJECT_DIR / ".env")
    load_env_file(REPO_DIR / ".env")


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _env_choice(name: str, default: str, choices: tuple) -> str:
    value = (os.environ.get(name) or default).strip().lower()
    if value not in choices:
        raise ConfigurationError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """Resolved deployment-wide configuration for one process."""

    var_dir: Path
    embedding_model: str
    secret_key: str
    admin_api_key: str
    default_top_k: int
    default_max_tokens: int
    history_db: str
    history_turns: int
    vector_backend: str
    qdrant_url: str
    qdrant_api_key: str

    @property
    def tenants_dir(self) -> Path:
        """One subdirectory per tenant: config, secrets, documents, index."""
        return self.var_dir / "tenants"

    def require_admin_key(self) -> str:
        if not self.admin_api_key:
            raise ConfigurationError(
                "ENTERPRISE_ADMIN_API_KEY is not set. Tenant management is "
                "disabled until an operator key is configured."
            )
        return self.admin_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the settings once per process (call ``get_settings.cache_clear()`` in tests)."""
    load_dotenv()
    return Settings(
        # Everything generated at runtime — tenant records, uploaded documents,
        # indexes — lives under var/ so it stays out of version control.
        var_dir=_env_path("ENTERPRISE_VAR_DIR", PROJECT_DIR / "var"),
        # One embedding model serves every tenant. Embeddings carry no tenant
        # secrets and the model is hundreds of megabytes, so sharing it is the
        # whole point — isolation happens at the index, not the encoder.
        embedding_model=os.environ.get("ENTERPRISE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        # Encrypts tenant provider keys at rest. Losing it means re-collecting
        # every tenant's provider key, so treat it like a database password.
        secret_key=os.environ.get("ENTERPRISE_SECRET_KEY", ""),
        # Operator credential for creating and listing tenants.
        admin_api_key=os.environ.get("ENTERPRISE_ADMIN_API_KEY", ""),
        default_top_k=_env_int("ENTERPRISE_TOP_K", 4),
        default_max_tokens=_env_int("ENTERPRISE_MAX_TOKENS", 1024),
        # Empty means in-process memory, which is fine for one dev server and
        # silently wrong under gunicorn's multiple workers. See history.py.
        history_db=os.environ.get("ENTERPRISE_HISTORY_DB", ""),
        history_turns=_env_int("ENTERPRISE_HISTORY_TURNS", 6),
        # Where vectors live: "faiss" keeps them in per-tenant files (default,
        # nothing extra to run); "qdrant" keeps them in per-tenant collections
        # on a Qdrant server. See vectordb.py for why both exist.
        vector_backend=_env_choice(
            "ENTERPRISE_VECTOR_BACKEND", "faiss", ("faiss", "qdrant")
        ),
        # ":memory:" runs Qdrant inside the process — used by the tests so the
        # suite needs no server; real deployments point at one.
        qdrant_url=os.environ.get("ENTERPRISE_QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_api_key=os.environ.get("ENTERPRISE_QDRANT_API_KEY", ""),
    )
