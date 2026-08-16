"""Command line access to the engine: ``python -m ragengine <command>``.

    python -m ragengine create-tenant "Acme Corp" --provider anthropic --provider-key sk-ant-...
    python -m ragengine list-tenants
    python -m ragengine --tenant acme-corp add-doc handbook.md ./handbook.md
    python -m ragengine --tenant acme-corp build-index
    python -m ragengine --tenant acme-corp search "refund policy"   # no API call, no cost
    python -m ragengine --tenant acme-corp ask "what is the refund policy?"
    python -m ragengine --tenant acme-corp chat

``search`` is the debugging tool: it prints exactly which passages were
retrieved and their distances, which separates a retrieval problem from a
generation problem.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from .exceptions import RagError
from .indexing import build_index, save_document
from .pipeline import get_pipeline
from .tenants import PROVIDERS, get_tenant_store
from .vectordb import index_location, index_stamp_path


def _require_tenant(args: argparse.Namespace):
    if not args.tenant:
        raise RagError("this command needs --tenant <slug>")
    return get_tenant_store().get(args.tenant)


def _cmd_create_tenant(args: argparse.Namespace) -> int:
    tenant, api_key = get_tenant_store().create(
        name=args.name,
        provider=args.provider,
        provider_api_key=args.provider_key,
        model=args.model,
        slug=args.slug,
    )
    print(f"Created tenant {tenant.slug!r} ({tenant.provider} / {tenant.model})")
    print(f"API key (shown once, store it now): {api_key}")
    return 0


def _cmd_list_tenants(args: argparse.Namespace) -> int:
    tenants = get_tenant_store().list()
    if not tenants:
        print("No tenants yet.")
        return 0
    for tenant in tenants:
        ready = "ready" if index_stamp_path(tenant).is_file() else "no index"
        print(f"  {tenant.slug:<24} {tenant.provider:<10} {tenant.model:<24} [{ready}]")
    return 0


def _cmd_add_doc(args: argparse.Namespace) -> int:
    tenant = _require_tenant(args)
    text = Path(args.path).read_text(encoding="utf-8", errors="replace")
    save_document(tenant, args.filename, text)
    store = build_index(tenant)
    print(f"Stored {args.filename} and reindexed: {len(store)} chunks")
    return 0


def _cmd_build_index(args: argparse.Namespace) -> int:
    tenant = _require_tenant(args)
    print(f"Reading documents from {tenant.documents_dir}")
    store = build_index(tenant)
    print(f"Indexed {len(store)} chunks -> {index_location(tenant)}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    tenant = _require_tenant(args)
    results = get_pipeline(tenant).retrieve(" ".join(args.question), args.top_k)
    if not results:
        print("No matches.")
        return 0
    for result in results:
        preview = result.text[:90].strip()
        print(f"  {result.distance:.3f}  {result.source:<16} {preview}...")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    tenant = _require_tenant(args)
    answer = get_pipeline(tenant).answer(" ".join(args.question), args.top_k)
    print(answer.text)
    print(
        f"\n[{answer.provider} · {answer.model} · "
        f"{answer.input_tokens} in / {answer.output_tokens} out]"
    )
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    tenant = _require_tenant(args)
    pipeline = get_pipeline(tenant)
    # One id for the session, so follow-ups can refer back to earlier answers.
    conversation_id = str(uuid.uuid4())
    print(f"Chatting with {tenant.name}'s knowledge base. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", ""}:
            return 0
        print("\n" + pipeline.answer(question, args.top_k, conversation_id).text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ragengine", description=__doc__.splitlines()[0])
    parser.add_argument("--tenant", help="tenant slug for per-tenant commands")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="how many chunks to retrieve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-tenant", help="onboard a new customer")
    create.add_argument("name")
    create.add_argument("--provider", required=True, choices=PROVIDERS)
    create.add_argument("--provider-key", required=True, help="the customer's own LLM API key")
    create.add_argument("--model", help="override the provider's default model")
    create.add_argument("--slug", help="override the generated slug")
    create.set_defaults(handler=_cmd_create_tenant)

    subparsers.add_parser("list-tenants", help="show every tenant").set_defaults(
        handler=_cmd_list_tenants
    )

    add_doc = subparsers.add_parser("add-doc", help="upload one document and reindex")
    add_doc.add_argument("filename", help="name to store it as, e.g. handbook.md")
    add_doc.add_argument("path", help="local file to read")
    add_doc.set_defaults(handler=_cmd_add_doc)

    subparsers.add_parser("build-index", help="rebuild the tenant's index").set_defaults(
        handler=_cmd_build_index
    )

    search = subparsers.add_parser("search", help="show retrieved chunks without calling the API")
    search.add_argument("question", nargs="+")
    search.set_defaults(handler=_cmd_search)

    ask = subparsers.add_parser("ask", help="answer a single question")
    ask.add_argument("question", nargs="+")
    ask.set_defaults(handler=_cmd_ask)

    subparsers.add_parser("chat", help="interactive question loop").set_defaults(handler=_cmd_chat)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except RagError as exc:
        # Expected failures (no tenant, no index) read better without a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
