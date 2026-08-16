# CLAUDE.md — enterprise_rag

This file provides guidance to Claude Code (claude.ai/code) when working in
the `enterprise_rag/` application. It supplements the repo-root CLAUDE.md.

## What this application is

A multi-tenant RAG service. Each tenant (customer) brings **their own
documents** and **their own LLM API key** — Anthropic or Google Gemini — and
gets an isolated FAISS knowledge base, a keyed JSON API, and a hosted chat
page at `/chat/<slug>/`. It is the multi-tenant evolution of `../ragbot/`;
where the two overlap, the conventions are deliberately identical, so read
the ragbot section of the root CLAUDE.md too.

## Commands

The virtualenv is at the **repo root** (`../.venv/`, Python 3.9). Run
everything from `enterprise_rag/` — `ragengine`, `ragapi` and `web` are
top-level packages here, which is what keeps imports free of path juggling.

```bash
cd enterprise_rag && source ../.venv/bin/activate
```

```bash
python manage.py test                          # 76 tests, < 1s, no network
python manage.py test tests.test_views         # one module
python manage.py test tests.test_views.AskApiTests.test_returns_the_answer
python manage.py runserver                     # http://127.0.0.1:8000

# CLI — same engine as the API, no HTTP involved
python -m ragengine create-tenant "Acme Corp" --provider anthropic --provider-key sk-ant-...
python -m ragengine list-tenants
python -m ragengine --tenant acme-corp add-doc handbook.md ./handbook.md
python -m ragengine --tenant acme-corp build-index
python -m ragengine --tenant acme-corp search "refund policy"   # retrieval only — no API call, no cost
python -m ragengine --tenant acme-corp ask "how do refunds work?"
python -m ragengine --tenant acme-corp chat
```

`search` is the debugging tool: it prints exactly which passages were
retrieved and their distances, which separates a retrieval problem from a
generation problem — always check it before blaming the model.

There is no linter or formatter configured, and no pytest — the suite runs on
Django's own runner with `SimpleTestCase`.

## Architecture

Three packages, one direction of knowledge:

- **`ragengine/`** — the engine. Contains **no Django imports anywhere**.
  Tenants, chunking, embeddings, FAISS store, providers, history, pipeline,
  CLI. The web app, the CLI and the tests all call the same functions.
- **`ragapi/`** — the HTTP layer. Knows nothing about embeddings. Validates
  input, authenticates, calls the engine, and maps `RagError` subclasses onto
  status codes. Also serves the per-tenant chat template.
- **`web/`** — Django project settings/urls/wsgi. `DATABASES = {}`.

Flow per question: `authenticate` (API key → tenant) → `get_pipeline`
(per-tenant cache) → `condense` (follow-up rewritten standalone, on the
tenant's own model, *for retrieval only*) → `retrieve` (tenant's FAISS index)
→ `complete` (tenant's provider, tenant's key) → history recorded under the
tenant's slug.

### The tenancy model

A tenant is a directory, not a database row: `var/tenants/<slug>/` holds
`tenant.json` (public config + API-key hash), `secrets.json` (provider key,
encrypted), `documents/`, `index/`. Everything under `var/` is gitignored.

- **Isolation is structural, not filtered.** Every tenant has their own index
  files and their own provider client. There is no shared index a bug could
  leak across. Never introduce one.
- **The tenant is always derived from the API key** (`erag.<slug>.<secret>`),
  never from a request field. Keep it that way — it is what makes cross-tenant
  access impossible even by accident.
- **The filesystem is the cross-worker coordination mechanism.** Tenant reads
  are mtime-cached (`tenants.py`), and pipelines are cached per slug keyed on
  the mtimes of `index.faiss` *and* `tenant.json` (`pipeline.py`). A reindex
  or config change done by one gunicorn worker or the CLI is picked up by all
  workers on the next request, with no restart and no signalling. Preserve
  this property when adding state.
- **One embedding model serves every tenant** (`embeddings.py`). Embeddings
  carry no secrets; sharing the encoder and isolating the indexes is the
  intended split. Per-tenant choice exists only for the *chat* model.

### The vector backend abstraction

`ragengine/vectordb.py` decides where vectors live, chosen by
`ENTERPRISE_VECTOR_BACKEND`: `faiss` (default — per-tenant `index.faiss` +
`chunks.json`, nothing extra to run) or `qdrant` (one collection per tenant,
`erag-<slug>`, on a Qdrant server). Callers use `save_index` / `load_store` /
`index_stamp_path` / `index_info`; nothing above vectordb branches on the
backend name. Qdrant rebuilds stamp `index/qdrant.json` so the
mtime-cache-coordination invariant survives a server-side backend, and the
stamp records the vector count so `load_store` refuses a drifted collection —
the same in-step contract as FAISS. Tests run Qdrant with
`ENTERPRISE_QDRANT_URL=:memory:` (in-process, no server) and are skipped when
`qdrant-client` is absent.

### The operator console

`/console/` is a static shell (no auth on the page; every data call sends
`X-Admin-Key`, so nothing leaks). It drives the admin API: list tenants with
vector counts, `GET /api/v1/tenants/<slug>/` for per-document chunk counts,
`POST /api/v1/tenants/<slug>/documents/` for operator uploads. Tenant API
keys are still shown exactly once, at creation, together with copyable
integration snippets.

### The provider abstraction

`ragengine/providers/base.py` defines the whole contract: a system prompt
plus neutral `{"role": "user"|"assistant", "content": str}` messages in, a
`Completion` out. `anthropic.py` uses the official `anthropic` SDK (no
LangChain here, unlike ragbot); `gemini.py` prefers `google-genai` and falls
back to legacy `google-generativeai`. Adding a provider = one module + one
line in `providers/__init__.py`. Nothing above the provider layer may import
an SDK or branch on provider name.

Conversation history is stored in that same neutral shape (`history.py`,
stdlib sqlite3 or in-process dict) precisely so a tenant can switch providers
mid-conversation.

## Invariants — do not break these

- **Index contract**: `index.faiss` row *i* corresponds to `chunks.json`
  entry *i*. `VectorStore.load()` refuses them out of step. Both are
  rebuildable, never committed.
- **Nothing loads at import time.** `faiss`, `sentence_transformers`,
  `anthropic`, `google.genai` are all imported inside functions/properties.
  This keeps `import ragengine` and the test suite fast. Preserve lazy
  imports when adding code.
- **All deployment config goes through `ragengine/config.py`** — no other
  module touches `os.environ`. Per-tenant config lives in the tenant record,
  not in env vars. `get_settings()` is `lru_cache`d; tests must call
  `get_settings.cache_clear()` (plus `get_embedder.cache_clear()` and
  `clear_pipelines()` — `EngineTestCase` does all three).
- **Never send `temperature`.** Current Claude models reject it with a 400.
  There is intentionally no temperature knob anywhere.
- **Anthropic default model is `claude-opus-5`**, Gemini's is
  `gemini-2.5-flash` (`tenants.DEFAULT_MODELS`). A refusal response
  (`stop_reason == "refusal"`) with empty content gets `REFUSAL_TEXT`, never
  an unchecked `content[0]`.
- **Two model calls per turn once a conversation exists**: condense (for
  retrieval), then answer (the user's original wording, history in context).
- **Grounding**: the system prompt tells the model to answer only from
  retrieved context or the conversation, otherwise say it doesn't know.
  That is what makes this RAG rather than a chatbot; don't loosen it casually.
- **No database.** Tenants are directories, history is sqlite3-by-path,
  tests use `SimpleTestCase`. Do not add Django models.
- **Secrets discipline**: tenant API keys are stored only as SHA-256 hashes
  (plaintext returned exactly once, at creation); provider keys are
  Fernet-encrypted under `ENTERPRISE_SECRET_KEY` (plaintext-0600 fallback is
  marked `"scheme": "plain"`); all comparisons are `hmac.compare_digest`.
  Nothing secret ever appears in `tenant.json`, `public_dict()`, or logs.
- **Fail closed**: the admin endpoint returns 503 when
  `ENTERPRISE_ADMIN_API_KEY` is unset, not open access.
- **Input validation lives at the boundary**: slugs (`SLUG_RE`), uploaded
  filenames (`FILENAME_RE`, no traversal), document size (2 MB), question
  length (2000 chars), conversation ids (UUID regex) — all checked before
  anything touches the filesystem or a store.

## HTTP conventions

- `/api/v1/*` is `csrf_exempt`: called cross-origin, protected by keys
  (`X-Admin-Key` for tenant management, `X-Api-Key` for everything else) plus
  the CORS allowlist. The chat page POST is same-origin and CSRF-protected,
  so it needs no key.
- Error mapping: validation → 400, bad key → 401, chat disabled → 403,
  duplicate tenant → 409, index not built → 409, provider failure → 502,
  engine unavailable → 503, unexpected → 500 (logged with `logger.exception`).
- `/api/v1/health/` deliberately loads no index and reads no tenant data, so
  it stays cheap to poll.
- Document upload reindexes synchronously — correct at knowledge-base scale
  because the next question must see the new content. If corpora ever get
  huge, move the rebuild to a queue; the API shape should not change.

## Tests

Nothing in the suite downloads a model, opens a network connection, or spends
money. Keep it that way:

- `tests/base.py` is the scaffolding: `EngineTestCase` points
  `ENTERPRISE_VAR_DIR` at a tempdir and clears every process cache;
  `FakeEmbedder` is a crc32 bag-of-words (real FAISS, deterministic
  similarity — **never use `hash()`**, Python randomizes it per process);
  `FakeProvider` records calls and returns canned text.
- Patch points: `ragengine.store.get_embedder` for retrieval,
  `ragengine.pipeline.build_provider` for pipelines,
  `ragapi.views.answer_question` for view tests.
- Provider tests inject a stub client (`provider._client = ...`) rather than
  patching the SDK import.

## Environment

`.env.example` lists every variable with notes on the ones that bite. The
ones people forget:

- `ENTERPRISE_ADMIN_API_KEY` — required before tenant management works at all.
- `ENTERPRISE_SECRET_KEY` — set it **before** onboarding the first tenant;
  changing it later means re-collecting every tenant's provider key.
- `ENTERPRISE_HISTORY_DB` — empty means in-process history, which is fine for
  `runserver` and silently wrong under gunicorn's multiple workers.
- There is deliberately **no** `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` at the
  deployment level — every tenant brings their own key at onboarding.

`google-genai` and `cryptography` are in `requirements.txt` but may be absent
from the shared venv; the code degrades gracefully (explicit
`ProviderNotInstalledError`, plaintext-0600 secrets) rather than crashing.

## Voice

The codebase is heavily commented in a specific voice: module docstrings
explain *why* a design choice was made, not what the code does. Match that
when adding modules.
