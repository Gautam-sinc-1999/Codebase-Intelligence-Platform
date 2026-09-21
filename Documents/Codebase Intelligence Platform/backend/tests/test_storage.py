"""Index persistence, incremental re-index and conversation durability. Covers F-13, F-15, F-23, F-37."""
import time

from app.core.database import FallbackMongoDBStore
from app.ingestion.index_store import RepositoryIndexStore, index_store
from app.vector.chromadb_client import chroma_store
import app.api.repositories as repositories


# --------------------------------------------------------------------- index persistence (F-15)

def test_index_survives_a_cold_cache(temp_repo):
    """
    The chunk cache was the only copy, so anything indexed by another process looked like an
    empty repository. It is now a cache in front of the on-disk index.
    """
    _, repo_id, chunks = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    assert index_store.exists(repo_id)

    repositories.repository_chunks_cache.pop(repo_id, None)
    reloaded = repositories.get_repository_chunks(repo_id)

    assert len(reloaded) == len(chunks)
    assert {c["symbol"] for c in reloaded} == {c["symbol"] for c in chunks}


def test_derived_content_is_rebuilt_not_stored_twice(temp_repo):
    _, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    raw = index_store.load(repo_id)
    assert all(c.get("formatted_content") for c in raw), "must be rebuilt on load"


def test_a_corrupt_index_does_not_prevent_startup(tmp_path):
    store = RepositoryIndexStore(str(tmp_path / "indexes"))
    (tmp_path / "indexes").mkdir(parents=True)
    (tmp_path / "indexes" / "broken.json").write_text("{ not valid json")
    assert store.load("broken") is None  # logged, not raised


def test_index_writes_are_atomic(tmp_path):
    store = RepositoryIndexStore(str(tmp_path / "indexes"))
    store.save("r", [{"chunk_id": "1", "symbol": "s", "file_path": "a.py",
                      "start_line": 1, "end_line": 2, "code_snippet": "x", "language": "python"}])
    leftovers = list((tmp_path / "indexes").glob("*.tmp"))
    assert not leftovers, f"interrupted-write temp files left behind: {leftovers}"


# --------------------------------------------------------------------- incremental (F-23)

def test_unchanged_reindex_reuses_stored_chunks(temp_repo, tmp_path):
    root, repo_id, _ = temp_repo({f"m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(25)})

    start = time.perf_counter()
    repositories.index_repository_folder(root, repo_id, repo_id)
    incremental = time.perf_counter() - start

    start = time.perf_counter()
    repositories.index_repository_folder(root, repo_id, repo_id, force_full=True)
    full = time.perf_counter() - start

    assert incremental < full, "unchanged re-index should be cheaper than a full rebuild"
    assert index_store.load_manifest(repo_id), "file hashes must be persisted to skip work"


def test_changed_file_is_picked_up(temp_repo):
    root, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    with open(f"{root}/a.py", "a") as handle:
        handle.write("\ndef newly_added():\n    return 2\n")

    result = repositories.index_repository_folder(root, repo_id, repo_id)
    assert "newly_added" in {c["symbol"] for c in result["all_chunks"]}


def test_deleted_file_is_removed_from_index_and_vector_store(temp_repo):
    import os
    root, repo_id, _ = temp_repo({
        "keep.py": "def keeper():\n    return 1\n",
        "drop.py": "def dropped_function():\n    return 2\n",
    })
    os.remove(f"{root}/drop.py")
    result = repositories.index_repository_folder(root, repo_id, repo_id)

    assert "dropped_function" not in {c["symbol"] for c in result["all_chunks"]}
    hits = chroma_store.search_similar(repo_id, "dropped_function", top_k=5)
    assert not [h for h in hits if h["metadata"].get("symbol") == "dropped_function"], \
        "deleted code is still retrievable and would be cited as if it existed"


# --------------------------------------------------------------------- divergence (F-37)

def test_reindex_repairs_a_drained_vector_store(temp_repo):
    """
    Incremental re-index only re-embeds changed files, so a vector store that lost its contents
    was never repaired: the chunk index looked complete while semantic search had nothing.
    """
    root, repo_id, chunks = temp_repo({"billing.py": "def invoice_total(items):\n    return sum(items)\n"})
    chroma_store.delete_chunks(repo_id, [c["chunk_id"] for c in chunks])
    assert chroma_store.get_or_create_collection(repo_id).count() == 0

    repositories.index_repository_folder(root, repo_id, repo_id)   # source unchanged
    assert chroma_store.get_or_create_collection(repo_id).count() >= len(chunks)


# --------------------------------------------------------------------- conversations (F-13)

async def test_conversations_persist_across_processes(tmp_path):
    """The fallback store advertised persistence while holding everything in a dict."""
    store = FallbackMongoDBStore(str(tmp_path / "fallback_db"))
    collection = store.get_collection("conversations")
    await collection.insert_one({
        "conversation_id": "c1", "repository_id": "r1", "title": "Where is X?",
        "messages": [{"role": "user", "content": "Where is X?"}],
        "working_context": {"current_symbol": "DiscountService"}, "summary": "investigating X",
    })

    # A second store over the same directory stands in for a restarted process.
    reopened = FallbackMongoDBStore(str(tmp_path / "fallback_db"))
    conversation = await reopened.get_collection("conversations").find_one({"conversation_id": "c1"})

    assert conversation is not None, "conversation lost on restart"
    assert conversation["title"] == "Where is X?"
    assert conversation["working_context"]["current_symbol"] == "DiscountService"
    assert len(conversation["messages"]) == 1


async def test_conversation_updates_are_persisted(tmp_path):
    store = FallbackMongoDBStore(str(tmp_path / "db"))
    collection = store.get_collection("conversations")
    await collection.insert_one({"conversation_id": "c2", "messages": []})
    await collection.update_one(
        {"conversation_id": "c2"},
        {"$push": {"messages": {"$each": [{"role": "user", "content": "hi"}]}},
         "$set": {"title": "hi"}},
    )

    reopened = FallbackMongoDBStore(str(tmp_path / "db"))
    conversation = await reopened.get_collection("conversations").find_one({"conversation_id": "c2"})
    assert conversation["title"] == "hi" and len(conversation["messages"]) == 1


def test_degraded_backends_are_reported(api_client):
    """Every store falls back silently; without this the system looks identical either way."""
    payload = api_client.get("/").json()
    assert "backends" in payload and "degraded" in payload
    assert set(payload["backends"]) == {"documents", "cache", "vectors", "graph"}


# --------------------------------------------------------------------- vector client lifecycle

def test_vector_client_reopens_after_close(temp_repo):
    """close() runs on shutdown, but the store is a module-level singleton reused afterwards."""
    _, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    before = len(chroma_store.search_similar(repo_id, "alpha", top_k=3))
    chroma_store.close()
    after = len(chroma_store.search_similar(repo_id, "alpha", top_k=3))
    assert after == before, "client did not reopen after close"
    chroma_store.close()  # repeated close must be safe
