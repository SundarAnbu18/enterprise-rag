"""Root URL configuration — everything is delegated to the ragapi app."""

from django.urls import include, path

urlpatterns = [
    path("", include("ragapi.urls")),
]
