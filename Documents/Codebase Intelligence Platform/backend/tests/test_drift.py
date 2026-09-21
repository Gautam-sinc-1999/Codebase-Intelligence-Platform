"""
Repository drift detection.

A thread is a conversation about a specific version of the code. These tests cover the three
states the check can report, and the one property that matters more than any of them: a thread
must remain usable when the check cannot be answered.
"""
import os
import subprocess
import time
import asyncio

import pytest

import app.api.github as github_module
from app.core.database import db_client
from app.core.jobs import indexing_jobs
from app.ingestion.git_source import clone_repository, remote_head_sha, _branch_cache
from app.memory.conversation_memory import ConversationMemory
from app.memory.mongo_thread_store import thread_store


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
    (root / "api.py").write_text("def handler(r):\n    return 1\n")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "initial", cwd=root)
    return root


@pytest.fixture(autouse=True)
def _clear_cache():
    _branch_cache.clear()
    yield
    _branch_cache.clear()


@pytest.fixture
def local_github(monkeypatch, origin_repo):
    monkeypatch.setattr(github_module, "clone_repository",
                        lambda url, branch, dest, **k: clone_repository(str(origin_repo), branch, dest, allow_local=True))
    monkeypatch.setattr(github_module, "list_remote_branches",
                        lambda url, **k: {"default_branch": "main",
                                          "branches": [{"name": "main", "commit_sha": "a" * 40, "is_default": True}]})
    monkeypatch.setattr(github_module, "remote_head_sha",
                        lambda url, branch, **k: remote_head_sha(str(origin_repo), branch, allow_local=True))
    return origin_repo


async def _settle(repository_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = indexing_jobs.get(repository_id)
        if job is not None and job.is_terminal:
            await asyncio.sleep(0.05)
            return job
        await asyncio.sleep(0.05)
    raise AssertionError("job did not settle")


async def _import(api_client, repo="widgets"):
    body = api_client.post("/api/repositories/from-github",
                           json={"url": f"https://github.com/acme/{repo}", "branch": "main"}).json()
    await _settle(body["repository_id"])
    return body["repository_id"]


async def _answered_thread(api_client, repository_id, conversation_id):
    """A thread with one real turn, so it carries an indexed_commit_sha."""
    await ConversationMemory.get_or_create_conversation(conversation_id, repository_id)
    await ConversationMemory.save_turn(conversation_id, "what does handler do?", {
        "answer": "It returns 1.",
        "sources": [{"file_path": "api.py", "symbol": "handler", "start_line": 1, "end_line": 2}],
    })
    return conversation_id


# ---------------------------------------------------------------- the three states

async def test_a_current_repository_reports_no_drift(api_client, local_github):
    repo_id = await _import(api_client, "current-repo")
    conv = await _answered_thread(api_client, repo_id, f"conv_cur_{repo_id}")

    body = api_client.get(f"/api/repositories/{repo_id}/drift",
                          params={"conversation_id": conv}).json()
    assert body["state"] == "current" and body["behind"] is False
    assert body["remote_sha"] == body["indexed_sha"]


async def test_an_upstream_commit_is_reported_as_behind(api_client, local_github):
    """The case in the brief: the branch moved on while the thread sat idle."""
    repo_id = await _import(api_client, "moving-repo")
    conv = await _answered_thread(api_client, repo_id, f"conv_behind_{repo_id}")

    (local_github / "new.py").write_text("def added():\n    return 2\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "upstream moved", cwd=local_github)
    _branch_cache.clear()

    body = api_client.get(f"/api/repositories/{repo_id}/drift",
                          params={"conversation_id": conv}).json()
    assert body["state"] == "behind" and body["behind"] is True
    assert body["remote_sha"] != body["indexed_sha"]


async def test_a_synced_thread_is_told_in_its_transcript_not_by_a_banner(api_client, local_github):
    """
    Sync from thread A and thread B is discussing a version that is no longer indexed.

    B is told — but by the marker written into its transcript at the exact point the version
    changed, which is a better place for it than a banner floating at the top of the thread. The
    marker advances B's recorded commit, so B is not *also* reported as drifted: having been told
    once is enough, and `thread_behind` is reserved for a thread that has not been told at all.
    """
    repo_id = await _import(api_client, "shared-repo")
    old_thread = await _answered_thread(api_client, repo_id, f"conv_old_{repo_id}")

    (local_github / "new.py").write_text("def added():\n    return 2\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "moved", cwd=local_github)
    _branch_cache.clear()

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)

    messages = await thread_store.read_messages(old_thread)
    markers = [m for m in messages if m.get("event") == "repository_synced"]
    assert len(markers) == 1, "the thread was not told its version changed"

    body = api_client.get(f"/api/repositories/{repo_id}/drift",
                          params={"conversation_id": old_thread}).json()
    assert body["state"] == "current"
    assert body["behind"] is False, "offered a sync that would do nothing"

    # The per-message stamps still differ, which is what keeps old citations honestly labelled.
    stamps = {m.get("commit_sha") for m in messages if m["role"] == "assistant"}
    assert stamps and stamps != {body["indexed_sha"]}


async def test_a_thread_the_index_moved_under_without_marking_is_reported_behind(
        api_client, local_github):
    """
    `thread_behind` covers the case the marker does not: the index moved by some route that did
    not record a boundary in the transcript — a zip re-index, a restored thread, a marking that
    failed. The thread is then discussing code nothing has told it about.
    """
    repo_id = await _import(api_client, "unmarked-repo")
    conv = await _answered_thread(api_client, repo_id, f"conv_unmarked_{repo_id}")

    # Simulates exactly that: the repository advances while the thread's record does not.
    await db_client.connect()
    await db_client.get_collection("repositories").update_one(
        {"repository_id": repo_id}, {"$set": {"commit_sha": "f" * 40}})

    body = api_client.get(f"/api/repositories/{repo_id}/drift",
                          params={"conversation_id": conv}).json()

    assert body["state"] == "thread_behind"
    assert body["behind"] is False, "offered a sync that would do nothing"
    assert body["thread_sha"] != body["indexed_sha"]
    assert "older version" in body["detail"]


# ---------------------------------------------------------------- robustness

async def test_a_zip_repository_is_never_behind_and_makes_no_network_call(api_client, temp_repo):
    _, repo_id, _ = temp_repo({"a.py": "def a():\n    return 1\n"})
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": repo_id, "name": "zipped", "storage_path": None})

    called = []
    original = github_module.remote_head_sha
    github_module.remote_head_sha = lambda *a, **k: called.append(1) or original(*a, **k)
    try:
        body = api_client.get(f"/api/repositories/{repo_id}/drift").json()
    finally:
        github_module.remote_head_sha = original

    assert body["state"] == "current" and body["behind"] is False
    assert called == [], "consulted the network for a repository with no upstream"


async def test_an_unreachable_remote_never_blocks_the_thread(api_client, local_github, monkeypatch):
    """
    The property that outranks the feature itself.

    Drift is advisory. If GitHub is down, the thread must still open and still answer — so the
    check reports 'unknown' rather than failing.
    """
    from app.ingestion.git_source import CloneFailed

    repo_id = await _import(api_client, "offline-repo")
    conv = await _answered_thread(api_client, repo_id, f"conv_off_{repo_id}")

    monkeypatch.setattr(github_module, "remote_head_sha",
                        lambda *a, **k: (_ for _ in ()).throw(CloneFailed("Could not resolve host")))

    response = api_client.get(f"/api/repositories/{repo_id}/drift",
                              params={"conversation_id": conv})
    assert response.status_code == 200, "an unreachable remote became a client-visible failure"
    body = response.json()
    assert body["state"] == "unknown" and body["behind"] is False
    assert "resolve host" in body["error"]

    # And the thread itself is unaffected.
    assert api_client.get(f"/api/conversations/{conv}").status_code == 200


async def test_a_deleted_branch_is_reported_rather_than_looking_current(api_client, local_github, monkeypatch):
    repo_id = await _import(api_client, "gone-branch-repo")
    monkeypatch.setattr(github_module, "remote_head_sha", lambda *a, **k: None)

    body = api_client.get(f"/api/repositories/{repo_id}/drift").json()
    assert body["state"] == "branch_gone"
    assert "no longer exists" in body["detail"]


def test_drift_on_an_unknown_repository_is_404(api_client):
    assert api_client.get("/api/repositories/nope/drift").status_code == 404


async def test_drift_without_a_conversation_still_reports_the_repository(api_client, local_github):
    """The sidebar asks about a repository; only a thread banner needs the thread comparison."""
    repo_id = await _import(api_client, "no-conv-repo")
    body = api_client.get(f"/api/repositories/{repo_id}/drift").json()
    assert body["state"] in ("current", "behind")
    assert body["thread_sha"] == ""


# ---------------------------------------------------------------- stamping

async def test_a_turn_records_the_commit_it_was_answered_against(api_client, local_github):
    repo_id = await _import(api_client, "stamped-repo")
    conv = await _answered_thread(api_client, repo_id, f"conv_stamp_{repo_id}")

    await db_client.connect()
    record = await db_client.get_collection("repositories").find_one({"repository_id": repo_id})

    meta = await thread_store.read_meta(conv)
    assert meta["indexed_commit_sha"] == record["commit_sha"]

    messages = await thread_store.read_messages(conv)
    assistant = [m for m in messages if m["role"] == "assistant"][-1]
    assert assistant["commit_sha"] == record["commit_sha"]


async def test_turns_either_side_of_a_sync_record_different_commits(api_client, local_github):
    """
    A long thread can straddle a sync, so the stamp lives on the message as well as the thread —
    the turns before describe different code from the turns after.
    """
    repo_id = await _import(api_client, "straddle-repo")
    conv = await _answered_thread(api_client, repo_id, f"conv_straddle_{repo_id}")

    (local_github / "new.py").write_text("def added():\n    return 2\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "moved", cwd=local_github)
    _branch_cache.clear()

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)

    await ConversationMemory.save_turn(conv, "and now?", {"answer": "Different.", "sources": []})

    stamps = [m.get("commit_sha") for m in await thread_store.read_messages(conv) if m["role"] == "assistant"]
    assert len(stamps) == 2
    assert stamps[0] != stamps[1], "both turns recorded the same commit across a sync"


async def test_a_zip_thread_carries_no_commit_stamp(api_client, temp_repo):
    """Nothing to stamp, and an empty string is not a version anyone should reason about."""
    _, repo_id, _ = temp_repo({"a.py": "def a():\n    return 1\n"})
    await db_client.connect()
    await db_client.get_collection("repositories").insert_one(
        {"repository_id": repo_id, "name": "zipped", "storage_path": None})

    conv = f"conv_zip_{repo_id}"
    await ConversationMemory.get_or_create_conversation(conv, repo_id)
    await ConversationMemory.save_turn(conv, "hi", {"answer": "hello", "sources": []})

    assert (await thread_store.read_meta(conv))["indexed_commit_sha"] == ""
    assistant = [m for m in await thread_store.read_messages(conv) if m["role"] == "assistant"][-1]
    assert "commit_sha" not in assistant


# ---------------------------------------------------------------- caching

async def test_the_drift_check_is_cached_but_sync_is_not(api_client, local_github):
    """
    Different tolerance for staleness, deliberately.

    Drift runs on every thread open and noticing a commit a few minutes late costs nothing.
    /sync uses the same primitive to decide whether to do any work at all, where a stale answer
    would skip a sync the user explicitly asked for.
    """
    repo_id = await _import(api_client, "cache-repo")

    calls = []
    real = remote_head_sha

    def counting(url, branch, **kwargs):
        calls.append(kwargs.get("use_cache", False))
        return real(str(local_github), branch, allow_local=True)

    github_module.remote_head_sha = counting
    try:
        api_client.get(f"/api/repositories/{repo_id}/drift")
        api_client.post(f"/api/repositories/{repo_id}/sync")
    finally:
        github_module.remote_head_sha = real

    assert calls[0] is True, "drift check bypassed the cache"
    assert calls[1] is False, "sync consulted a cache that could hide a real change"
