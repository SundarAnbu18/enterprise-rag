"""Request authentication: operator key for management, tenant key for use.

Two credentials, two audiences. ``X-Admin-Key`` is the operator's — it can
create and list tenants and nothing else. ``X-Api-Key`` is a tenant's — it
scopes every request to exactly that tenant's documents, index, provider key
and history; the tenant is *derived from the key*, never from a request field,
so a tenant cannot name another tenant even by accident.
"""

from __future__ import annotations

import hmac
from functools import wraps

from django.http import HttpRequest, JsonResponse

from ragengine import AuthenticationError, get_settings, get_tenant_store


def require_admin(view):
    """Gate a view behind the operator key.

    Refuses everything when no key is configured — an unset credential must
    fail closed, not open.
    """

    @wraps(view)
    def wrapper(request: HttpRequest, *args, **kwargs):
        expected = get_settings().admin_api_key
        if not expected:
            return JsonResponse(
                {"error": "tenant management is disabled: ENTERPRISE_ADMIN_API_KEY is not set"},
                status=503,
            )
        presented = request.headers.get("X-Admin-Key", "")
        # Constant-time compare so the key can't be guessed a byte at a time.
        if not hmac.compare_digest(presented, expected):
            return JsonResponse({"error": "unauthorized"}, status=401)
        return view(request, *args, **kwargs)

    return wrapper


def require_tenant(view):
    """Resolve the tenant from ``X-Api-Key`` and pass it into the view."""

    @wraps(view)
    def wrapper(request: HttpRequest, *args, **kwargs):
        try:
            tenant = get_tenant_store().authenticate(request.headers.get("X-Api-Key", ""))
        except AuthenticationError:
            return JsonResponse({"error": "unauthorized"}, status=401)
        return view(request, tenant, *args, **kwargs)

    return wrapper
