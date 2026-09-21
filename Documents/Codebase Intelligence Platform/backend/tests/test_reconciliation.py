"""
Startup reconciliation of repository records (G-15).

`DELETE` cleans up properly, but nothing reconciled what was already there. A record whose index
file *and* source directory had both vanished — a manually cleared `data/`, a restored database, a
half-finished migration — still appeared in the sidebar as a selectable repository and failed on
first use.

The interesting half is what must be **kept**: a GitHub import has no `storage_path` by design,
so treating a missing directory as fatal would delete exactly the repositories that are easiest
to restore.
"""
import os

import pytest

from app.core.database import db_client
from app.ingestion.index_store import index_store
from app.api.repositories import reconcile_repositories


async def _record(**fields):
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(fields)


async def _ids():
    await db_client.connect()
    rows = await db_client.get_collection("repositories").find({}).to_list(length=1000)
    return {r["repository_id"] for r in rows}


# ------------------------------------------------------------------ removed

async def test_a_record_with_no_index_and_no_source_is_removed(api_client):
    await _record(repository_id="ghost_repo", name="ghost", storage_path="/gone/for/good")

    result = await reconcile_repositories()

    assert "ghost_repo" in result["removed"]
    assert "ghost_repo" not in await _ids()


async def test_a_record_with_no_storage_path_at_all_is_removed(api_client):
    """Two of the records in the original session had no `storage_path` recorded."""
    await _record(repository_id="pathless_repo", name="pathless")

    await reconcile_repositories()
    assert "pathless_repo" not in await _ids()


async def test_removal_also_purges_anything_it_left_behind(api_client, temp_repo):
    """A record that goes should not leave embeddings and graph nodes nothing can reach."""
    from app.graph.neo4j_client import neo4j_client

    _, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    await _record(repository_id=repo_id, name="derived", storage_path="/gone")

    # Make it unusable by removing only the index, leaving the derived stores populated.
    index_store.delete(repo_id)

    await reconcile_repositories()

    assert repo_id not in await _ids()
    if neo4j_client.is_fallback:
        remaining = sum(1 for n in neo4j_client.fallback.g.nodes
                        if neo4j_client.fallback._repo(n) == repo_id)
    else:
        with neo4j_client.driver.session() as session:
            remaining = session.run(
                "MATCH (x) WHERE x.repo_id = $id RETURN count(x) AS n", id=repo_id).single()["n"]
    assert remaining == 0, "graph nodes survived the record that owned them"


# ------------------------------------------------------------------ kept

async def test_a_record_with_a_persisted_index_is_kept(api_client, temp_repo):
    _, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    await _record(repository_id=repo_id, name="indexed", storage_path="/gone/but/indexed")

    assert index_store.exists(repo_id), "precondition: the index is on disk"
    await reconcile_repositories()
    assert repo_id in await _ids()


async def test_a_record_whose_source_still_exists_is_kept(api_client, tmp_path):
    """No index yet, but it can be re-indexed from what is on disk."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text("def alpha():\n    return 1\n")

    await _record(repository_id="reindexable", name="reindexable", storage_path=str(source))

    await reconcile_repositories()
    assert "reindexable" in await _ids()


async def test_a_github_record_is_kept_even_with_no_index_and_no_source(api_client):
    """
    The case that makes a naive rule destructive. A GitHub import deletes its clone after
    indexing, so it has no `storage_path` by design — and `/sync` can re-acquire it in seconds.
    Deleting it would throw away the URL and branch, which is the only thing that cannot be
    regenerated.
    """
    await _record(repository_id="gh_recoverable", name="acme/widgets@main",
                  source="github", clone_url="https://github.com/acme/widgets.git",
                  branch="main")

    result = await reconcile_repositories()

    assert "gh_recoverable" not in result["removed"]
    assert "gh_recoverable" in await _ids()


async def test_a_github_record_without_a_clone_url_is_not_recoverable(api_client):
    """`source: github` alone is not a route back — the URL is what makes it recoverable."""
    await _record(repository_id="gh_broken", name="broken", source="github")

    await reconcile_repositories()
    assert "gh_broken" not in await _ids()


# ------------------------------------------------------------------ shape and safety

async def test_reconciliation_reports_what_it_did(api_client, temp_repo):
    _, keeper, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    await _record(repository_id=keeper, name="keeper", storage_path="/gone")
    await _record(repository_id="doomed_repo", name="doomed", storage_path="/also/gone")

    result = await reconcile_repositories()

    assert result["checked"] >= 2
    assert result["kept"] >= 1
    assert "doomed_repo" in result["removed"] and keeper not in result["removed"]


async def test_reconciliation_is_idempotent(api_client):
    await _record(repository_id="twice_repo", name="twice", storage_path="/gone")

    first = await reconcile_repositories()
    second = await reconcile_repositories()

    assert "twice_repo" in first["removed"]
    assert "twice_repo" not in second["removed"], "removed a record that was already gone"


async def test_an_unreachable_database_is_not_treated_as_an_empty_one(api_client, monkeypatch):
    """
    A database that cannot be read must never be mistaken for a database with no repositories —
    that reading would delete nothing here, but the function must not claim it checked anything.
    """
    import app.api.repositories as repos_module

    class Exploding:
        def find(self, *args, **kwargs):
            raise RuntimeError("database unreachable")

    monkeypatch.setattr(repos_module.db_client, "get_collection", lambda name: Exploding())

    result = await reconcile_repositories()
    assert result["checked"] == 0 and result["removed"] == []
