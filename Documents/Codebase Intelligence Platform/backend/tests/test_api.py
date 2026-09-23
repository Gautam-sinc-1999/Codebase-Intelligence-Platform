"""HTTP surface: auth, CORS, upload safety, reindex, streaming. Covers F-03, F-17, F-18, F-22, F-33, F-40."""
import io
import json
import zipfile

import pytest

from app.core.config import settings
from app.api.repositories import _safe_extract_zip, _is_ignorable_archive_entry, UnsafeArchiveError

PROTECTED_ROUTES = ["/api/repositories", "/api/conversations", "/api/graph/anything"]


def _zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


def _write_zip(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return str(path)


# --------------------------------------------------------------------- archive safety (F-03)

@pytest.mark.parametrize("entry", [
    "../../evil.txt",
    "../../../../etc/passwd",
    "/tmp/absolute-evil.txt",
])
def test_path_traversal_is_rejected(tmp_path, entry):
    archive = _write_zip(tmp_path / "slip.zip", [(entry, "pwned")])
    with pytest.raises(UnsafeArchiveError, match="escapes"):
        _safe_extract_zip(archive, str(tmp_path / "out"))
    assert not (tmp_path / "evil.txt").exists()


def test_benign_archive_extracts(tmp_path):
    archive = _write_zip(tmp_path / "ok.zip", [("proj/app.py", "def f(): pass"), ("proj/sub/b.py", "x=1")])
    _safe_extract_zip(archive, str(tmp_path / "out"))
    assert (tmp_path / "out" / "proj" / "app.py").exists()


def test_entry_count_limit_applies_to_indexable_entries(tmp_path):
    over = settings.MAX_ARCHIVE_ENTRIES + 1
    archive = _write_zip(tmp_path / "many.zip", [(f"f{i}.txt", "x") for i in range(over)])
    with pytest.raises(UnsafeArchiveError, match="exceeding the limit"):
        _safe_extract_zip(archive, str(tmp_path / "out"))


# --------------------------------------------------------------------- archive filtering (F-40)

def test_ignorable_entries_are_filtered_before_counting(tmp_path):
    """
    A repository zipped with its virtualenv was rejected outright: 22,827 of 22,841 entries were
    in ignored directories and only 14 were indexable.
    """
    entries = [(f"venv/lib/site-packages/mod{i}.py", "x") for i in range(settings.MAX_ARCHIVE_ENTRIES + 500)]
    entries += [("__MACOSX/._thing", "x"), ("proj/app.py", "def real(): pass")]
    archive = _write_zip(tmp_path / "bulky.zip", entries)

    _safe_extract_zip(archive, str(tmp_path / "out"))          # must not raise
    extracted = list((tmp_path / "out").rglob("*.py"))
    assert [p.name for p in extracted] == ["app.py"]


def test_secrets_are_never_written_to_disk(tmp_path):
    """Skipped at extraction, not merely at indexing: we should never store others' credentials."""
    archive = _write_zip(tmp_path / "secret.zip", [
        ("proj/app.py", "def f(): pass"),
        ("proj/.env", "GROQ_API_KEY=sk-live-SECRET"),
        ("proj/certs/server.pem", "-----BEGIN PRIVATE KEY-----"),
        ("proj/.env.example", "GROQ_API_KEY="),
    ])
    out = tmp_path / "out"
    _safe_extract_zip(archive, str(out))

    written = {p.name for p in out.rglob("*") if p.is_file()}
    assert ".env" not in written and "server.pem" not in written
    assert ".env.example" in written, "templates are safe and useful"


@pytest.mark.parametrize("name,ignorable", [
    ("venv/lib/x.py", True), ("node_modules/pkg/i.js", True), ("__MACOSX/._x", True),
    (".git/config", True), ("proj/.env", True), ("proj/key.pem", True),
    ("proj/app.py", False), ("proj/.env.example", False), (".github/workflows/ci.yml", False),
])
def test_archive_entry_classification(name, ignorable):
    assert _is_ignorable_archive_entry(name) is ignorable


# --------------------------------------------------------------------- auth (F-17)

def test_routes_are_open_when_no_api_key_is_configured(api_client):
    settings.API_KEY = ""
    for route in PROTECTED_ROUTES:
        assert api_client.get(route).status_code != 401


def test_routes_require_the_key_when_configured(api_client):
    settings.API_KEY = "s3cret-value"
    try:
        for route in PROTECTED_ROUTES:
            assert api_client.get(route).status_code == 401
            assert api_client.get(route, headers={"X-API-Key": "s3cret-value"}).status_code != 401
            assert api_client.get(route, headers={"Authorization": "Bearer s3cret-value"}).status_code != 401
            assert api_client.get(route, headers={"X-API-Key": "wrong"}).status_code == 401

        assert api_client.post("/api/repositories/upload",
                               files={"file": ("x.zip", b"PK", "application/zip")}).status_code == 401
        assert api_client.delete("/api/conversations/any").status_code == 401
    finally:
        settings.API_KEY = ""


def test_health_hides_backend_detail_from_anonymous_callers(api_client):
    settings.API_KEY = "s3cret-value"
    try:
        anonymous = api_client.get("/").json()
        assert anonymous["auth"] == "enabled"
        assert "backends" not in anonymous, "backend detail is reconnaissance once exposed"
        authenticated = api_client.get("/", headers={"X-API-Key": "s3cret-value"}).json()
        assert "backends" in authenticated
    finally:
        settings.API_KEY = ""


# --------------------------------------------------------------------- CORS (F-18)

def test_cors_allows_configured_origins_only(api_client):
    allowed = api_client.get("/api/repositories", headers={"Origin": "http://localhost:3000"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert allowed.headers.get("access-control-allow-credentials") == "true"

    denied = api_client.get("/api/repositories", headers={"Origin": "https://evil.example.com"})
    assert not denied.headers.get("access-control-allow-origin")


def test_wildcard_origin_disables_credentials():
    """The two cannot legally combine; browsers reject the pairing outright."""
    original = settings.CORS_ORIGINS
    try:
        settings.CORS_ORIGINS = "*"
        assert settings.cors_allows_any_origin
    finally:
        settings.CORS_ORIGINS = original


# --------------------------------------------------------------------- conversations (F-22)

async def _make_conversation(repository_id, conversation_id):
    from app.core.database import db_client
    from app.memory.conversation_memory import ConversationMemory
    await db_client.connect()
    await ConversationMemory.get_or_create_conversation(conversation_id, repository_id, "t")


async def test_unindexed_repository_reports_state_not_a_claim_about_the_code(api_client):
    """
    "This repository is not indexed" was presented as "I couldn't find that in your code" — a
    claim about the user's codebase rather than about our own state.
    """
    from app.core.database import db_client
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": "unindexed_repo", "name": "my-app"})
    await _make_conversation("unindexed_repo", "conv_unindexed")

    response = api_client.post("/api/conversations/conv_unindexed/messages", json={"message": "hi"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "my-app" in detail and "indexed" in detail


async def test_missing_repository_is_404(api_client):
    await _make_conversation("repo_that_vanished", "conv_missing")
    response = api_client.post("/api/conversations/conv_missing/messages", json={"message": "hi"})
    assert response.status_code == 404


def test_unknown_conversation_is_404_on_both_endpoints(api_client):
    for path in ["/api/conversations/nope/messages", "/api/conversations/nope/messages/stream"]:
        assert api_client.post(path, json={"message": "x"}).status_code == 404


# --------------------------------------------------------------------- streaming (F-33)

async def test_streaming_emits_meta_then_tokens_then_done(api_client, sample_repo_path):
    from app.core.database import db_client
    from app.api.repositories import index_repository_folder

    index_repository_folder(sample_repo_path, "stream_repo", "sample")
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": "stream_repo", "name": "sample", "storage_path": sample_repo_path})
    await _make_conversation("stream_repo", "conv_stream")

    events = []
    with api_client.stream("POST", "/api/conversations/conv_stream/messages/stream",
                           json={"message": "Where is the discount calculation?"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        for line in response.iter_lines():
            if line and line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))

    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta"
    # Sources arrive before generation so the UI can render citations immediately.
    assert events[0]["sources"]

    # `done` carries the answer and is emitted *before* the turn is written, so a failure to
    # persist costs the reader their record of the answer but never the answer itself. `saved`
    # therefore trails it, carrying the position the answer was written at — which is the only
    # way the reader can rate an answer they just watched arrive without reloading first.
    assert kinds[-2:] == ["done", "saved"]
    done = events[-2]
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == done["answer"]
    assert isinstance(events[-1]["answer_seq"], int)


async def test_streamed_turn_is_persisted(api_client):
    conversation = api_client.get("/api/conversations/conv_stream").json()
    roles = [m["role"] for m in conversation.get("messages", [])]
    assert roles[:2] == ["user", "assistant"]


# --------------------------------------------------------------------- reindex (F-32)

async def test_reindex_and_status_endpoints(api_client, tmp_path):
    from app.core.database import db_client
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text("def alpha():\n    return 1\n")

    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": "rx", "name": "my-repo", "storage_path": str(source)})

    assert api_client.get("/api/repositories/rx/status").json()["status"] == "not indexed"

    first = api_client.post("/api/repositories/rx/reindex").json()
    assert first["status"] == "ready" and first["mode"] == "incremental"

    (source / "b.py").write_text("def beta():\n    return alpha()\n")
    assert api_client.post("/api/repositories/rx/reindex").json()["file_count"] == 2
    assert api_client.post("/api/repositories/rx/reindex?full=true").json()["mode"] == "full"
    assert api_client.get("/api/repositories/rx/status").json()["index_persisted"] is True


def test_reindex_errors_are_specific(api_client):
    assert api_client.post("/api/repositories/does-not-exist/reindex").status_code == 404


async def test_stored_citations_resolve_back_to_source(api_client, tmp_path):
    """
    A persisted turn holds citation pointers, not code bodies (see ConversationMemory).

    That trade only works if a pointer can be turned back into source on demand, otherwise a
    resumed thread shows citations whose code can never be opened — which is exactly what the
    UI's code viewer does when a reader expands one.
    """
    from app.core.database import db_client

    source = tmp_path / "src"
    source.mkdir()
    (source / "billing.py").write_text(
        "def outer():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def calculate_total(items):\n"
        "    subtotal = sum(i.price for i in items)\n"
        "    return subtotal\n"
    )

    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": "cite", "name": "cite-repo", "storage_path": str(source)})
    api_client.post("/api/repositories/cite/reindex")

    def snippet(**params):
        return api_client.get("/api/repositories/cite/snippet", params=params)

    # Exact pointer — the common case, a citation replayed verbatim from storage.
    exact = snippet(file_path="billing.py", start_line=5, end_line=7).json()
    assert exact["symbol"] == "calculate_total"
    assert "subtotal = sum(i.price for i in items)" in exact["code"]
    assert exact["start_line"] == 5

    # A pointer landing inside a symbol still resolves to that symbol, not to a neighbour.
    inside = snippet(file_path="billing.py", start_line=6, end_line=7).json()
    assert inside["symbol"] == "calculate_total"

    # Errors distinguish "no index" from "not in this repository" — the two have different fixes.
    assert snippet(file_path="nope.py", start_line=1, end_line=2).status_code == 404
    missing = api_client.get("/api/repositories/never_indexed/snippet",
                             params={"file_path": "billing.py", "start_line": 1, "end_line": 2})
    assert missing.status_code == 409


async def test_a_resumed_thread_can_open_every_citation_it_stored(api_client, tmp_path):
    """The end-to-end shape of the trade: stored turns carry no code, yet every one opens."""
    from app.core.database import db_client
    from app.memory.conversation_memory import ConversationMemory

    source = tmp_path / "src2"
    source.mkdir()
    (source / "cart.py").write_text("def add_item(cart, item):\n    cart.append(item)\n    return cart\n")

    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": "resume", "name": "resume-repo", "storage_path": str(source)})
    api_client.post("/api/repositories/resume/reindex")

    await ConversationMemory.get_or_create_conversation("conv_resume", "resume")
    await ConversationMemory.save_turn("conv_resume", "how are items added?", {
        "answer": "add_item appends to the cart.",
        "sources": [{
            "file_path": "cart.py", "symbol": "add_item", "start_line": 1, "end_line": 3,
            "entity_type": "function",
            # Present in the live response, deliberately dropped on the way to disk.
            "code_snippet": "def add_item(cart, item):\n    cart.append(item)\n    return cart",
        }],
    })

    stored = api_client.get("/api/conversations/conv_resume").json()
    citations = [m for m in stored["messages"] if m["role"] == "assistant"][-1]["sources"]
    assert citations and all("code_snippet" not in c for c in citations)

    for citation in citations:
        resolved = api_client.get(
            f"/api/repositories/{stored['repository_id']}/snippet",
            params={"file_path": citation["file_path"],
                    "start_line": citation["start_line"],
                    "end_line": citation["end_line"]},
        )
        assert resolved.status_code == 200
        assert resolved.json()["code"].strip()
