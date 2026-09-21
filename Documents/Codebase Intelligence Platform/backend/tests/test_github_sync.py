"""
Syncing a GitHub repository.

The claim under test is that a sync is cheap: a fresh clone is an entirely new directory, yet
only genuinely changed files are re-parsed and re-embedded. That works because the manifest keys
on content hash rather than path or mtime — so these tests assert reuse counts, not just that
the endpoint returned 200.
"""
import os
import subprocess
import time
import asyncio

import pytest

import app.api.github as github_module
from app.core.database import db_client
from app.core.jobs import indexing_jobs
from app.ingestion.git_source import clone_repository, remote_head_sha
from app.ingestion.index_store import index_store


def _git(*args, cwd):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    return subprocess.run(["git", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True)


@pytest.fixture
def origin_repo(tmp_path):
    root = tmp_path / "origin"
    root.mkdir()
    _git("init", "-b", "main", cwd=root)
    (root / "api.py").write_text("def handler(r):\n    return total(r)\n")
    (root / "calc.py").write_text("def total(r):\n    return 1\n")
    (root / "util.py").write_text("def helper():\n    return 2\n")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "initial", cwd=root)
    return root


@pytest.fixture
def local_github(monkeypatch, origin_repo):
    """Cloning and SHA lookup point at the local repository; everything else is the real path."""
    def fake_clone(url, branch, dest, **kwargs):
        return clone_repository(str(origin_repo), branch, dest, allow_local=True)

    def fake_branches(url, **kwargs):
        return {"default_branch": "main",
                "branches": [{"name": "main", "commit_sha": "a" * 40, "is_default": True}]}

    def fake_head(url, branch, **kwargs):
        return remote_head_sha(str(origin_repo), branch, allow_local=True)

    monkeypatch.setattr(github_module, "clone_repository", fake_clone)
    monkeypatch.setattr(github_module, "list_remote_branches", fake_branches)
    monkeypatch.setattr(github_module, "remote_head_sha", fake_head)
    return origin_repo


async def _settle(repository_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = indexing_jobs.get(repository_id)
        if job is not None and job.is_terminal:
            await asyncio.sleep(0.05)
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {repository_id} did not settle")


async def _import(api_client):
    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets"}).json()
    await _settle(body["repository_id"])
    return body["repository_id"]


# ---------------------------------------------------------------- no-op sync

async def test_sync_reports_already_current_without_cloning(api_client, local_github):
    """One ls-remote is much cheaper than a clone and a re-index; say so and stop."""
    repo_id = await _import(api_client)

    cloned = []
    original = github_module.clone_repository
    github_module.clone_repository = lambda *a, **k: cloned.append(1) or original(*a, **k)
    try:
        response = api_client.post(f"/api/repositories/{repo_id}/sync")
    finally:
        github_module.clone_repository = original

    assert response.status_code == 200
    assert response.json()["status"] == "already current"
    assert cloned == [], "cloned despite the branch not having moved"


async def test_force_full_syncs_even_when_current(api_client, local_github):
    """`full=true` is the escape hatch for when the index itself is suspect, not the source."""
    repo_id = await _import(api_client)

    response = api_client.post(f"/api/repositories/{repo_id}/sync?full=true")
    assert response.status_code == 202
    assert response.json()["mode"] == "full"

    job = await _settle(repo_id)
    assert job.state == "done"
    assert job.result["reused_files"] == 0, "a full sync should re-parse everything"


# ---------------------------------------------------------------- real drift

async def test_sync_picks_up_an_upstream_commit(api_client, local_github):
    repo_id = await _import(api_client)
    before = api_client.get(f"/api/repositories/{repo_id}/status").json()["commit_sha"]

    (local_github / "feature.py").write_text("def brand_new():\n    return 42\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "add feature", cwd=local_github)

    response = api_client.post(f"/api/repositories/{repo_id}/sync")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "syncing"
    assert body["from_commit"] == before and body["to_commit"] != before

    await _settle(repo_id)

    after = api_client.get(f"/api/repositories/{repo_id}/status").json()
    assert after["commit_sha"] != before
    assert "brand_new" in {c["symbol"] for c in index_store.load(repo_id)}


async def test_sync_re_embeds_only_what_changed(api_client, local_github):
    """
    The claim that makes sync affordable.

    A fresh clone shares no paths or timestamps with the previous one, so reuse is possible only
    because the manifest compares content hashes.
    """
    repo_id = await _import(api_client)

    (local_github / "calc.py").write_text("def total(r):\n    return 999\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "change one file", cwd=local_github)

    api_client.post(f"/api/repositories/{repo_id}/sync")
    job = await _settle(repo_id)

    assert job.result["parsed_files"] == 1, "re-parsed more than the file that changed"
    assert job.result["reused_files"] == 2, "failed to reuse the untouched files"


async def test_sync_removes_code_that_was_deleted_upstream(api_client, local_github):
    """Deleted code must stop being retrievable, not linger in the vector store."""
    repo_id = await _import(api_client)
    assert "helper" in {c["symbol"] for c in index_store.load(repo_id)}

    os.remove(local_github / "util.py")
    _git("add", "-A", cwd=local_github)
    _git("commit", "-m", "drop util", cwd=local_github)

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)

    assert "helper" not in {c["symbol"] for c in index_store.load(repo_id)}


async def test_the_clone_is_deleted_after_a_sync(api_client, local_github):
    repo_id = await _import(api_client)

    (local_github / "x.py").write_text("def x():\n    return 1\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "x", cwd=local_github)

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)
    assert not os.path.exists(github_module._clone_dir(repo_id))


# ---------------------------------------------------------------- refusals

def test_syncing_an_unknown_repository_is_404(api_client):
    assert api_client.post("/api/repositories/nope/sync").status_code == 404


async def test_syncing_a_zip_repository_is_refused_and_points_at_reindex(api_client, temp_repo):
    """A zip has no upstream; the advice has to name the endpoint that does apply."""
    _, repo_id, _ = temp_repo({"a.py": "def a():\n    return 1\n"})
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": repo_id, "name": "zipped", "storage_path": None})

    response = api_client.post(f"/api/repositories/{repo_id}/sync")
    assert response.status_code == 409
    assert "reindex" in response.json()["detail"]


async def test_reindexing_a_github_repository_points_at_sync(api_client, local_github):
    """
    The symmetric case, and the reason it matters: the sidebar's re-index button would otherwise
    fail on every GitHub repository with advice to re-upload something never uploaded.
    """
    repo_id = await _import(api_client)

    response = api_client.post(f"/api/repositories/{repo_id}/reindex")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "sync" in detail and "re-clones" in detail
    assert "Upload the repository again" not in detail


async def test_a_deleted_upstream_branch_is_reported_as_gone(api_client, local_github, monkeypatch):
    repo_id = await _import(api_client)
    monkeypatch.setattr(github_module, "remote_head_sha", lambda *a, **k: None)

    response = api_client.post(f"/api/repositories/{repo_id}/sync")
    assert response.status_code == 410
    assert "no longer exists" in response.json()["detail"]


async def test_an_unreachable_remote_is_a_502_not_a_crash(api_client, local_github, monkeypatch):
    from app.ingestion.git_source import CloneFailed

    repo_id = await _import(api_client)

    def unreachable(*args, **kwargs):
        raise CloneFailed("Could not resolve host.")

    monkeypatch.setattr(github_module, "remote_head_sha", unreachable)

    response = api_client.post(f"/api/repositories/{repo_id}/sync")
    assert response.status_code == 502


async def test_a_failed_sync_leaves_the_previous_index_intact(api_client, local_github, monkeypatch):
    """
    Unlike a failed import, a failed sync must be survivable.

    The repository already worked; a network hiccup is no reason to destroy a usable index and
    leave the user with nothing.
    """
    from app.ingestion.git_source import CloneFailed

    repo_id = await _import(api_client)
    symbols_before = {c["symbol"] for c in index_store.load(repo_id)}

    (local_github / "y.py").write_text("def y():\n    return 1\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "y", cwd=local_github)

    monkeypatch.setattr(github_module, "clone_repository",
                        lambda *a, **k: (_ for _ in ()).throw(CloneFailed("network died")))

    api_client.post(f"/api/repositories/{repo_id}/sync")
    job = await _settle(repo_id)
    assert job.state == "failed"

    await db_client.connect()
    assert await db_client.get_collection("repositories").find_one(
        {"repository_id": repo_id}) is not None, "a failed sync deleted a working repository"
    assert index_store.exists(repo_id)
    assert {c["symbol"] for c in index_store.load(repo_id)} == symbols_before

    status = api_client.get(f"/api/repositories/{repo_id}/status").json()
    assert "network died" in status["error"]
