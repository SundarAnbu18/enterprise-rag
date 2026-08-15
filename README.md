# Enterprise RAG

A multi-tenant retrieval-augmented generation service. Each customer (tenant)
brings **their own documents** and **their own LLM key** — Anthropic or Google
Gemini — and gets an isolated knowledge base, a JSON API, and a ready-made
chat page at `/chat/<slug>/`.

It is the production-shaped evolution of the single-tenant `ragbot/` next
door: same engine/web split, same FAISS + sentence-transformers retrieval,
same "the CLI, the tests and the web app all reach one engine" philosophy —
generalised so one deployment serves any number of companies.

## How multi-tenancy works

```
                        ┌────────────────────────────────────────────┐
   X-Admin-Key ───────▶ │  /api/v1/tenants/      onboard customers   │
                        └────────────────────────────────────────────┘
                        ┌────────────────────────────────────────────┐
   X-Api-Key  ────────▶ │  /api/v1/documents/    upload + reindex    │
   (erag.<slug>.<secret>│  /api/v1/reindex/                          │
    — tenant derived    │  /api/v1/ask/          question in, answer │
    from the key)       └────────────────────────────────────────────┘
                        ┌────────────────────────────────────────────┐
   browser ───────────▶ │  /chat/<slug>/         hosted chat box     │
                        └────────────────────────────────────────────┘

var/tenants/<slug>/
├── tenant.json     provider, model, limits, API-key hash
├── secrets.json    the tenant's LLM key, Fernet-encrypted at rest
├── documents/      their uploaded corpus (.txt / .md)
└── index/          their FAISS index + chunks.json
```

Isolation is structural, not filtered: every tenant has their own index files,
their own documents directory, their own provider client built from their own
decrypted key, and conversation history rows tagged with their slug. There is
no shared index that a bug could leak across.

One **embedding model** (MiniLM) is shared by all tenants — embeddings carry
no secrets and the model costs hundreds of megabytes, so sharing the encoder
and isolating the indexes is the right split.

## Quick start

```bash
cd enterprise_rag && source ../.venv/bin/activate
pip install -r requirements.txt        # adds google-genai + cryptography
cp .env.example .env                   # set ENTERPRISE_ADMIN_API_KEY at least
python manage.py runserver             # http://127.0.0.1:8000
```

Onboard a customer (their key is shown exactly once — store it):

```bash
curl -s http://127.0.0.1:8000/api/v1/tenants/ \
  -H "X-Admin-Key: $ENTERPRISE_ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp", "provider": "anthropic", "provider_api_key": "sk-ant-..."}'
# → {"slug": "acme-corp", "api_key": "erag.acme-corp.…", "chat_url": "/chat/acme-corp/", ...}
```

A Gemini customer differs only in two fields:

```bash
  -d '{"name": "Globex", "provider": "gemini", "provider_api_key": "AIza..."}'
```

Give them knowledge and ask:

```bash
curl -s http://127.0.0.1:8000/api/v1/documents/ \
  -H "X-Api-Key: erag.acme-corp.…" -H "Content-Type: application/json" \
  -d '{"filename": "handbook.md", "text": "Refunds are issued within 14 days.\n\nShipping is free."}'

curl -s http://127.0.0.1:8000/api/v1/ask/ \
  -H "X-Api-Key: erag.acme-corp.…" -H "Content-Type: application/json" \
  -d '{"question": "how do refunds work?"}'
```

Or just open **http://127.0.0.1:8000/chat/acme-corp/** — the hosted chat box,
branded per tenant, with conversation memory and source chips.

## CLI

Everything the API does, from the shell (same engine, no HTTP):

```bash
python -m ragengine create-tenant "Acme Corp" --provider anthropic --provider-key sk-ant-...
python -m ragengine list-tenants
python -m ragengine --tenant acme-corp add-doc handbook.md ./handbook.md
python -m ragengine --tenant acme-corp search "refund policy"   # retrieval only — no API call
python -m ragengine --tenant acme-corp ask "how do refunds work?"
python -m ragengine --tenant acme-corp chat
```

`search` is the debugging tool: it shows exactly which passages were retrieved
and their distances, separating a retrieval problem from a generation problem.

## Architecture

The split that matters: **`ragengine` contains no Django imports anywhere**,
and `ragapi` knows nothing about embeddings. The views validate input, call
the engine, and map `RagError` subclasses onto status codes.

Per question: `authenticate` (key → tenant) → `get_pipeline` (cached per
tenant, keyed on index + config mtimes, so a reindex or provider switch is
picked up with no restart) → `condense` (follow-ups rewritten standalone, for
retrieval only, on the tenant's own model) → `retrieve` (their FAISS index)
→ `complete` (their provider, their key) → history recorded under their slug.

Providers implement one small contract (`ragengine/providers/base.py`): a
system prompt plus neutral `{"role", "content"}` messages in, a `Completion`
out. Adding a provider is one module and one registry line.

Invariants carried over from `ragbot` (and still enforced):

- **Index contract**: `index.faiss` row *i* is `chunks.json` entry *i*;
  `VectorStore.load()` refuses them out of step.
- **Nothing loads at import time** — SDKs, FAISS and the embedder are imported
  in functions; heavyweight objects are cached per process.
- **All deployment config goes through `ragengine/config.py`**; per-tenant
  config lives in the tenant record. Tests call `get_settings.cache_clear()`.
- **No temperature is ever sent** — current Claude models reject it with a 400.
- **Grounding**: the system prompt tells the model to answer only from the
  retrieved context or the conversation, and otherwise say it doesn't know.
- **No database** (`DATABASES = {}`): tenants are directories, history is
  SQLite-by-DSN, tests use `SimpleTestCase`.

## Security model

| Secret | Storage | Notes |
|---|---|---|
| Operator admin key | env only | fails **closed** when unset |
| Tenant API key | SHA-256 hash only | plaintext shown once at onboarding; slug rides in the key (`erag.<slug>.<secret>`) so lookup is O(1); compare is constant-time |
| Tenant provider key | Fernet-encrypted under `ENTERPRISE_SECRET_KEY` | falls back to a 0600 plain file (marked `"scheme": "plain"`) when `cryptography` is absent — fine locally, set the secret in production |

Uploaded filenames are strictly validated (no path traversal), documents are
size-capped, conversation ids must be UUIDs, and slugs are shape-checked
before ever touching the filesystem.

`/api/v1/*` endpoints are `csrf_exempt` because they are called cross-origin
with per-tenant keys plus the CORS allowlist; the chat page POST is
same-origin and CSRF-protected. The hosted chat page is public for any tenant
with `chat_enabled` (the default) — a tenant who wants it private disables it
and calls the API with their key instead.

## Tests

```bash
python manage.py test          # 76 tests, < 1s
```

Nothing in the suite downloads a model or spends money: the embedder is a
deterministic bag-of-words fake (real FAISS, real similarity), providers are
stubs, and `answer_question` is mocked at the view seam.

## Production notes

- Run under gunicorn behind nginx exactly like `ragbot` (see
  `docs/deployment.md` at the repo root); the WSGI module is `web.wsgi`.
- Set `ENTERPRISE_HISTORY_DB` — the in-memory history default is wrong under
  multiple workers.
- Set `ENTERPRISE_SECRET_KEY` **before** onboarding the first tenant; changing
  it later means re-collecting provider keys.
- Reindexing is synchronous by design at knowledge-base scale; the pipeline
  cache picks up a rebuilt index on the next request via file mtimes, across
  all workers, with no signalling.
# enterprise-rag
