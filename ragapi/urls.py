"""URLs for the enterprise RAG API and per-tenant chat pages."""

from django.urls import path

from . import views

urlpatterns = [
    path("api/v1/tenants/", views.tenants, name="tenants"),
    path("api/v1/documents/", views.upload_document, name="upload_document"),
    path("api/v1/reindex/", views.reindex, name="reindex"),
    path("api/v1/ask/", views.ask, name="ask"),
    path("api/v1/health/", views.health, name="health"),
    path("chat/<slug:slug>/", views.chat, name="chat"),
]
