"""The HTTP layer: a JSON API for machines, a chat page per tenant for humans.

These views hold no retrieval logic. They validate input, call ``ragengine``
and map ``RagError`` subclasses onto status codes — the pipeline itself is in
the package, where the CLI and the tests reach it too.

The API endpoints are ``csrf_exempt`` because they are called cross-origin by
tenants' own systems and widgets; they are protected by per-tenant API keys
(constant-time compared) plus the CORS allowlist instead. The chat page POST
is same-origin and CSRF-protected, so it needs no key.
"""

from __future__ import annotations

import json
import logging

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ragengine import (
    ConfigurationError,
    IndexNotBuiltError,
    NoDocumentsError,
    ProviderError,
    RagError,
    Tenant,
    TenantExistsError,
    TenantNotFoundError,
    answer_question,
    build_index,
    get_settings,
    get_tenant_store,
    is_valid_conversation_id,
    save_document,
)

from .auth import require_admin, require_tenant

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 2000

GENERIC_ERROR = "Something went wrong answering that. Please try again."
UNAVAILABLE_ERROR = "The assistant is unavailable right now."
BAD_CONVERSATION_ERROR = "'conversation_id' must be a UUID"


def _json_body(request: HttpRequest):
    """Parse and shape-check the request body, or return an error response."""
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JsonResponse({"error": "invalid JSON body"}, status=400)
    if not isinstance(payload, dict):
        return None, JsonResponse({"error": "body must be a JSON object"}, status=400)
    return payload, None


# ---------------------------------------------------------------------------
# Tenant management (operator key).
# ---------------------------------------------------------------------------


@csrf_exempt
@require_admin
def tenants(request: HttpRequest) -> JsonResponse:
    """POST onboards a customer; GET lists them."""
    store = get_tenant_store()

    if request.method == "GET":
        return JsonResponse({"tenants": [t.public_dict() for t in store.list()]})

    if request.method != "POST":
        return JsonResponse({"error": "method not allowed"}, status=405)

    payload, error = _json_body(request)
    if error:
        return error

    try:
        tenant, api_key = store.create(
            name=str(payload.get("name", "")),
            provider=str(payload.get("provider", "")),
            provider_api_key=str(payload.get("provider_api_key", "")),
            model=payload.get("model"),
            slug=payload.get("slug"),
        )
    except TenantExistsError as exc:
        return JsonResponse({"error": str(exc)}, status=409)
    except ConfigurationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    # The one and only time the tenant key crosses the wire from our side.
    body = tenant.public_dict()
    body["api_key"] = api_key
    body["chat_url"] = f"/chat/{tenant.slug}/"
    return JsonResponse(body, status=201)


# ---------------------------------------------------------------------------
# Tenant self-service (tenant key).
# ---------------------------------------------------------------------------


@csrf_exempt
@require_POST
@require_tenant
def upload_document(request: HttpRequest, tenant: Tenant) -> JsonResponse:
    """Add or replace one document in the tenant's corpus and reindex.

    Reindexing is synchronous: at knowledge-base scale it takes well under a
    second, and it means the very next question sees the new content. A
    deployment with huge corpora would move this to a queue — the API shape
    would not change.
    """
    payload, error = _json_body(request)
    if error:
        return error

    try:
        save_document(tenant, str(payload.get("filename", "")), str(payload.get("text", "")))
        store = build_index(tenant)
    except (ConfigurationError, NoDocumentsError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except RagError:
        logger.exception("Reindex failed for tenant %s", tenant.slug)
        return JsonResponse({"error": UNAVAILABLE_ERROR}, status=503)

    return JsonResponse({"status": "indexed", "chunks": len(store)}, status=201)


@csrf_exempt
@require_POST
@require_tenant
def reindex(request: HttpRequest, tenant: Tenant) -> JsonResponse:
    """Rebuild the tenant's index from whatever documents are on file."""
    try:
        store = build_index(tenant)
    except NoDocumentsError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except RagError:
        logger.exception("Reindex failed for tenant %s", tenant.slug)
        return JsonResponse({"error": UNAVAILABLE_ERROR}, status=503)
    return JsonResponse({"status": "indexed", "chunks": len(store)})


def _validated_question(payload: dict):
    """Pull (question, conversation_id) out of a payload, or an error response."""
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, None, JsonResponse({"error": "'question' is required"}, status=400)
    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        return None, None, JsonResponse(
            {"error": f"'question' must be at most {MAX_QUESTION_LENGTH} characters"},
            status=400,
        )

    conversation_id = payload.get("conversation_id") or ""
    if not isinstance(conversation_id, str):
        return None, None, JsonResponse({"error": BAD_CONVERSATION_ERROR}, status=400)
    conversation_id = conversation_id.strip()
    if conversation_id and not is_valid_conversation_id(conversation_id):
        return None, None, JsonResponse({"error": BAD_CONVERSATION_ERROR}, status=400)

    return question, conversation_id, None


def _answer_response(tenant: Tenant, question: str, conversation_id: str) -> JsonResponse:
    """Run the pipeline and translate failures into status codes."""
    try:
        answer = answer_question(tenant, question, conversation_id=conversation_id or None)
    except IndexNotBuiltError:
        return JsonResponse(
            {"error": "No documents have been indexed for this workspace yet."}, status=409
        )
    except ProviderError as exc:
        logger.warning("Provider failure for tenant %s: %s", tenant.slug, exc)
        return JsonResponse({"error": str(exc)}, status=502)
    except RagError:
        logger.exception("RAG pipeline unavailable for tenant %s", tenant.slug)
        return JsonResponse({"error": UNAVAILABLE_ERROR}, status=503)
    except Exception:
        logger.exception("Unexpected failure answering for tenant %s", tenant.slug)
        return JsonResponse({"error": GENERIC_ERROR}, status=500)

    body = answer.to_dict()
    if conversation_id:
        # Echo the conversation back so the client knows the turn was recorded.
        body["conversation_id"] = conversation_id
    return JsonResponse(body)


@csrf_exempt
@require_POST
@require_tenant
def ask(request: HttpRequest, tenant: Tenant) -> JsonResponse:
    """JSON endpoint tenants (and their embedded widgets) post questions to."""
    payload, error = _json_body(request)
    if error:
        return error
    question, conversation_id, error = _validated_question(payload)
    if error:
        return error
    return _answer_response(tenant, question, conversation_id)


# ---------------------------------------------------------------------------
# The hosted chat box (same-origin, CSRF-protected).
# ---------------------------------------------------------------------------


def chat(request: HttpRequest, slug: str) -> HttpResponse:
    """A ready-made chat page for one tenant at /chat/<slug>/.

    The page posts back here in the background so it can show a loader and
    keep the thread on screen. It is open to anyone who has the URL when the
    tenant's ``chat_enabled`` flag is on; a tenant who wants it private turns
    the flag off and calls /api/v1/ask/ with their key instead.
    """
    try:
        tenant = get_tenant_store().get(slug)
    except TenantNotFoundError:
        return HttpResponse("No such workspace.", status=404)
    if not tenant.chat_enabled:
        return HttpResponse("Chat is disabled for this workspace.", status=403)

    if request.method == "POST":
        payload, error = _json_body(request)
        if error:
            return error
        question, conversation_id, error = _validated_question(payload)
        if error:
            return error
        return _answer_response(tenant, question, conversation_id)

    return render(request, "ragapi/chat.html", {"tenant": tenant})


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    """Liveness check. Deliberately loads no index and counts no tenants' data."""
    tenants_dir = get_settings().tenants_dir
    return JsonResponse({"status": "ok", "tenants_dir": tenants_dir.is_dir()})
