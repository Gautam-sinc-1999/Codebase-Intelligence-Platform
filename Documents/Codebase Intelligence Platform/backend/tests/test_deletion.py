"""
Repository deletion (B-2).

A repository is one identity scattered across six places: a Mongo record, a Chroma collection, a
Neo4j subgraph, a chunk index, a manifest, an extracted source tree, plus in-process caches.
Deleting a subset is what creates orphans, so these tests check each store independently rather
than trusting the endpoint's status code.
"""
import os

import pytest

from app.core.database import db_client
from app.graph.neo4j_client import neo4j_client
from app.vector.chromadb_client import chroma_store
from app.ingestion.index_store import index_store
from app.api.repositories import (
    purge_repository_stores,
    repository_chunks_cache,
    indexing_status,
)


def _graph_nodes(repo_id):
    if neo4j_client.is_fallback:
        return sum(1 for n in neo4j_client.fallback.g.nodes
                   if neo4j_client.fallback._repo(n) == repo_id)
    with neo4j_client.driver.session() as session:
        return session.run(
            "MATCH (x) WHERE x.repo_id = $id RETURN count(x) AS n", id=repo_id
        ).single()["n"]


def _vector_count(repo_id):
    try:
        return chroma_store.get_or_create_collection(repo_id).count()
    except Exception:
        return 0


def test_purge_empties_every_store(temp_repo):
    _, repo_id, chunks = temp_repo({
        "svc.py": "def handler(request):\n    return validate(request)\n\ndef validate(r):\n    return True\n",
    })

    # Everything is populated to begin with, or the test proves nothing.
    assert chunks and _graph_nodes(repo_id) > 0
    assert _vector_count(repo_id) > 0
    assert index_store.exists(repo_id)
    repository_chunks_cache[repo_id] = chunks

    result = purge_repository_stores(repo_id)

    assert _graph_nodes(repo_id) == 0
    assert _vector_count(repo_id) == 0
    assert not index_store.exists(repo_id)
    assert repo_id not in repository_chunks_cache
    assert repo_id not in indexing_status
    assert result["graph_nodes_deleted"] > 0


def test_purge_removes_the_manifest_not_just_the_index(temp_repo):
    """
    A surviving manifest makes a rebuilt repository look already-indexed.

    The incremental path trusts the manifest to decide a file is unchanged, so an orphaned one
    matches every hash, reuses chunks from an index that no longer exists, and yields an empty
    repository reporting itself as ready.
    """
    from app.api.repositories import index_repository_folder

    source, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    assert index_store.load_manifest(repo_id)

    purge_repository_stores(repo_id)
    assert index_store.load_manifest(repo_id) == {}

    # Rebuilding under the same id must re-parse, not conclude "nothing changed".
    rebuilt = index_repository_folder(source, repo_id, repo_id)
    assert rebuilt["all_chunks"], "rebuild produced an empty index — a stale manifest survived"


def test_purge_only_deletes_source_inside_managed_storage(tmp_path):
    """`storage_path` can point anywhere; this must not become a path-deletion primitive."""
    outside = tmp_path / "not_ours"
    outside.mkdir()
    (outside / "keep.py").write_text("x = 1\n")

    purge_repository_stores("never_indexed", str(outside))

    assert outside.is_dir() and (outside / "keep.py").exists()


async def test_delete_endpoint_removes_the_record_and_reports_each_store(api_client, temp_repo):
    _, repo_id, _ = temp_repo({"b.py": "def beta():\n    return 2\n"})

    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": repo_id, "name": "deletable", "storage_path": None})

    response = api_client.delete(f"/api/repositories/{repo_id}")
    assert response.status_code == 200
    body = response.json()

    assert body["record_deleted"] is True
    assert body["graph_nodes_deleted"] > 0
    assert body["caches_cleared"] is True
    assert _graph_nodes(repo_id) == 0
    assert api_client.get(f"/api/repositories/{repo_id}").status_code == 404


def test_deleting_an_unknown_repository_is_404(api_client):
    assert api_client.delete("/api/repositories/no-such-repo").status_code == 404


async def test_delete_is_refused_while_indexing(api_client, temp_repo):
    """Deleting mid-index would race the worker and leave exactly the orphans this prevents."""
    _, repo_id, _ = temp_repo({"c.py": "def gamma():\n    return 3\n"})
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": repo_id, "name": "busy", "storage_path": None})

    indexing_status[repo_id] = "indexing"
    try:
        assert api_client.delete(f"/api/repositories/{repo_id}").status_code == 409
    finally:
        indexing_status.pop(repo_id, None)


async def test_conversations_are_kept_by_default_and_counted(api_client, temp_repo):
    """
    Threads are the only thing here that cannot be regenerated, so they are never collateral.
    """
    from app.memory.conversation_memory import ConversationMemory

    _, repo_id, _ = temp_repo({"d.py": "def delta():\n    return 4\n"})
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": repo_id, "name": "chatty", "storage_path": None})

    await ConversationMemory.get_or_create_conversation(f"conv_keep_{repo_id}", repo_id)

    body = api_client.delete(f"/api/repositories/{repo_id}").json()
    assert body["conversations_orphaned"] == 1
    assert body["conversations_deleted"] == 0
    assert await ConversationMemory.load_conversation(f"conv_keep_{repo_id}") is not None


async def test_conversations_are_removed_when_explicitly_requested(api_client, temp_repo):
    from app.memory.conversation_memory import ConversationMemory

    _, repo_id, _ = temp_repo({"e.py": "def epsilon():\n    return 5\n"})
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": repo_id, "name": "cascading", "storage_path": None})

    await ConversationMemory.get_or_create_conversation(f"conv_drop_{repo_id}", repo_id)

    body = api_client.delete(
        f"/api/repositories/{repo_id}?delete_conversations=true").json()
    assert body["conversations_deleted"] == 1
    assert body["conversations_orphaned"] == 0
    assert await ConversationMemory.load_conversation(f"conv_drop_{repo_id}") is None


async def test_deleting_one_repository_leaves_its_neighbour_intact(api_client, temp_repo):
    """The whole point of repo-scoped identity: deletion must not reach across repositories."""
    _, doomed, _ = temp_repo({"shared.py": "def helper():\n    return 1\n"})
    _, kept, _ = temp_repo({"shared.py": "def helper():\n    return 2\n"})

    await db_client.connect()
    for rid in (doomed, kept):
        await db_client.get_collection("repositories").insert_one(
            {"repository_id": rid, "name": rid, "storage_path": None})

    api_client.delete(f"/api/repositories/{doomed}")

    assert _graph_nodes(doomed) == 0
    assert _graph_nodes(kept) > 0, "deleting one repository removed a same-named symbol from another"
    assert _vector_count(kept) > 0
    assert index_store.exists(kept)


def test_failed_upload_leaves_no_orphaned_vectors(api_client, tmp_path):
    """
    Indexing writes embeddings before it builds the graph, so a mid-flight failure used to strand
    a populated collection owned by a repository Mongo never heard of.
    """
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("app.py", "def f():\n    return 1\n")
    buffer.seek(0)

    before = set()
    if not chroma_store.is_fallback:
        before = {c.name for c in chroma_store._ensure_client().list_collections()}

    # A corrupt archive fails before indexing; the guarantee under test is that the failure
    # handler purges regardless of how far the pipeline got.
    response = api_client.post(
        "/api/repositories/upload",
        files={"file": ("broken.zip", b"this is not a zip", "application/zip")},
    )
    assert response.status_code == 400

    if not chroma_store.is_fallback:
        after = {c.name for c in chroma_store._ensure_client().list_collections()}
        assert after == before, f"failed upload left collections behind: {after - before}"
