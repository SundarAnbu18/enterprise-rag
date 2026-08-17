"""Django settings for the enterprise RAG service.

Everything that differs between a laptop and a production VM comes from the
environment, so the same checkout runs in both places. Local defaults are the
safe ones; production overrides them through the systemd unit or
enterprise_rag/.env.
"""

from __future__ import annotations

import os
from pathlib import Path

from corsheaders.defaults import default_headers

from ragengine.config import load_dotenv

# enterprise_rag/web/settings.py -> enterprise_rag/
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()


def _env_list(name: str) -> list:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _env_flag(name: str, default: str = "False") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-secret-key-change-me")

DEBUG = _env_flag("DJANGO_DEBUG", "True")

ALLOWED_HOSTS = _env_list("DJANGO_ALLOWED_HOSTS") or (
    ["localhost", "127.0.0.1", "[::1]"] if DEBUG else []
)

# Needed for the chat form once the site is served over HTTPS behind a proxy.
CSRF_TRUSTED_ORIGINS = _env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "corsheaders",
    "ragapi",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "web.wsgi.application"
ASGI_APPLICATION = "web.asgi.application"

# No models, no sessions, no admin — tenants live on disk under var/tenants/,
# so this project genuinely has no database to configure.
DATABASES = {}

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# The site(s) allowed to embed a tenant chat widget cross-origin.
CORS_ALLOWED_ORIGINS = _env_list("CORS_ALLOWED_ORIGINS")
CORS_ALLOW_HEADERS = [*default_headers, "x-api-key", "x-admin-key"]

# Set DJANGO_EMBED_CHAT=True (HTTPS deployments only) when tenant sites iframe
# the chat page. Inside a cross-site iframe the browser only sends the CSRF
# cookie if it is SameSite=None + Secure; without this flag the embedded
# widget renders but every POST fails the CSRF check.
if _env_flag("DJANGO_EMBED_CHAT", "False"):
    CSRF_COOKIE_SAMESITE = "None"
    CSRF_COOKIE_SECURE = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "{levelname} {name}: {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
    },
    # Application logs go to stdout, which is where journalctl reads them from.
    "root": {"handlers": ["console"], "level": os.environ.get("DJANGO_LOG_LEVEL", "INFO")},
}
