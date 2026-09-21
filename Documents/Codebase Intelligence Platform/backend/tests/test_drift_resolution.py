"""
Resolving drift: acknowledging it, recording it, and telling the truth about aged citations.

The important test here is `test_a_citation_from_before_a_sync_is_flagged_stale`. Persisted turns
hold citation *pointers*, so after a sync an old pointer still resolves — against different code.
Showing that confidently is worse than showing nothing.
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
    (root / "billing.py").write_text(
        "def charge(amount):\n"
        "    return amount\n"
        "\n"
        "def refund(amount):\n"
        "    return -amount\n"
    )
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


async def _import(api_client, repo):
    body = api_client.post("/api/repositories/from-github",
                           json={"url": f"https://github.com/acme/{repo}", "branch": "main"}).json()
    await _settle(body["repository_id"])
    return body["repository_id"]


# ---------------------------------------------------------------- stale citations

async def test_a_citation_from_before_a_sync_is_flagged_stale(api_client, local_github):
    """
    The consequence of storing pointers instead of code.

    `billing.py:1-2` still resolves after the file changes — it just resolves to whatever is on
    those lines now. The viewer has to be told.
    """
    repo_id = await _import(api_client, "stale-repo")
    await db_client.connect()
    record = await db_client.get_collection("repositories").find_one({"repository_id": repo_id})
    cited_sha = record["commit_sha"]

    # Insert a function above the cited one, so everything below shifts.
    (local_github / "billing.py").write_text(
        "def audit(event):\n"
        "    return event\n"
        "\n"
        "def charge(amount):\n"
        "    return amount\n"
        "\n"
        "def refund(amount):\n"
        "    return -amount\n"
    )
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "insert audit above charge", cwd=local_github)
    _branch_cache.clear()

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)

    fresh = api_client.get(f"/api/repositories/{repo_id}/snippet",
                           params={"file_path": "billing.py", "start_line": 1, "end_line": 2}).json()
    assert fresh["stale"] is False, "a request with no cited commit should not claim staleness"

    aged = api_client.get(f"/api/repositories/{repo_id}/snippet",
                          params={"file_path": "billing.py", "start_line": 1,
                                  "end_line": 2, "at_sha": cited_sha}).json()
    assert aged["stale"] is True
    assert aged["cited_sha"] == cited_sha
    assert aged["indexed_sha"] != cited_sha
    # Lines 1-2 used to be `charge`; they are now `audit`.
    assert aged["symbol"] == "audit"


async def test_a_citation_at_the_current_commit_is_not_stale(api_client, local_github):
    repo_id = await _import(api_client, "fresh-repo")
    await db_client.connect()
    record = await db_client.get_collection("repositories").find_one({"repository_id": repo_id})

    body = api_client.get(f"/api/repositories/{repo_id}/snippet",
                          params={"file_path": "billing.py", "start_line": 1,
                                  "end_line": 2, "at_sha": record["commit_sha"]}).json()
    assert body["stale"] is False


def test_a_zip_repository_never_reports_staleness(api_client, temp_repo):
    """Nothing to compare against, so a cited sha cannot make a zip citation stale."""
    _, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})
    body = api_client.get(f"/api/repositories/{repo_id}/snippet",
                          params={"file_path": "a.py", "start_line": 1,
                                  "end_line": 2, "at_sha": "b" * 40}).json()
    assert body["stale"] is False


# ---------------------------------------------------------------- sync markers

async def test_a_sync_records_a_version_boundary_in_the_thread(api_client, local_github):
    repo_id = await _import(api_client, "marked-repo")
    conv = f"conv_marked_{repo_id}"
    await ConversationMemory.get_or_create_conversation(conv, repo_id)
    await ConversationMemory.save_turn(conv, "what does charge do?", {
        "answer": "It returns the amount.",
        "sources": [{"file_path": "billing.py", "symbol": "charge", "start_line": 1, "end_line": 2}],
    })

    (local_github / "billing.py").write_text("def charge(amount):\n    return amount * 2\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "double it", cwd=local_github)
    _branch_cache.clear()

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)

    messages = await thread_store.read_messages(conv)
    markers = [m for m in messages if m.get("event") == "repository_synced"]
    assert len(markers) == 1
    assert markers[0]["role"] == "system"
    assert "Synced to" in markers[0]["content"]
    assert markers[0]["from_commit"] != markers[0]["to_commit"]

    # The thread's own stamp advances, so it is no longer reported as left behind.
    assert (await thread_store.read_meta(conv))["indexed_commit_sha"] == markers[0]["to_commit"]


async def test_every_thread_on_the_repository_is_marked_not_just_one(api_client, local_github):
    """All of them cite code that moved, so all of them need the boundary."""
    repo_id = await _import(api_client, "multi-thread-repo")

    for n in (1, 2):
        conv = f"conv_multi{n}_{repo_id}"
        await ConversationMemory.get_or_create_conversation(conv, repo_id)
        await ConversationMemory.save_turn(conv, f"question {n}", {"answer": "a", "sources": []})

    (local_github / "billing.py").write_text("def charge(a):\n    return a + 1\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "change", cwd=local_github)
    _branch_cache.clear()

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)

    for n in (1, 2):
        messages = await thread_store.read_messages(f"conv_multi{n}_{repo_id}")
        assert any(m.get("event") == "repository_synced" for m in messages), f"thread {n} unmarked"


async def test_an_empty_thread_is_not_marked(api_client, local_github):
    """There is nothing above the line to qualify."""
    repo_id = await _import(api_client, "empty-thread-repo")
    conv = f"conv_empty_{repo_id}"
    await ConversationMemory.get_or_create_conversation(conv, repo_id)

    (local_github / "billing.py").write_text("def charge(a):\n    return a\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "change", cwd=local_github)
    _branch_cache.clear()

    api_client.post(f"/api/repositories/{repo_id}/sync")
    await _settle(repo_id)

    assert await thread_store.read_messages(conv) == []


async def test_markers_never_reach_the_language_model(api_client, local_github):
    """
    A marker is for the reader, not the model.

    It is also filtered *before* the history window is sliced, so a thread that has survived
    several syncs does not lose real turns to bookkeeping.
    """
    from app.agents.orchestrator import CodebaseAgentOrchestrator as AgentOrchestrator

    history = (
        [{"role": "user", "content": f"q{i}"} for i in range(3)]
        + [{"role": "system", "event": "repository_synced", "content": "Synced to abc1234"}]
        + [{"role": "assistant", "content": "a final answer"}]
    )

    built = AgentOrchestrator._build_messages("sys", "now what?", history)
    contents = [m["content"] for m in built]

    assert not any("Synced to" in c for c in contents), "a sync marker was sent to the model"
    assert "a final answer" in contents


def test_history_filtering_happens_before_the_window_is_applied():
    """
    Markers must not push real turns out of the window that exists to preserve them.

    Interleaved deliberately. With every marker bunched at the front, slicing before filtering
    would drop them anyway and the test would pass against either ordering — it has to be
    arranged so that only the correct order keeps a full window of real turns.
    """
    from app.agents.orchestrator import CodebaseAgentOrchestrator as AgentOrchestrator

    limit = AgentOrchestrator.MAX_HISTORY_MESSAGES
    history = []
    for i in range(limit):
        history.append({"role": "user", "content": f"turn{i}"})
        history.append({"role": "system", "event": "repository_synced", "content": "Synced"})

    built = AgentOrchestrator._build_messages("sys", "now?", history)
    kept = [m["content"] for m in built if m["content"].startswith("turn")]
    assert len(kept) == limit, (
        f"only {len(kept)} of {limit} real turns survived — markers consumed history slots"
    )


# ---------------------------------------------------------------- acknowledgement

async def test_acknowledging_drift_silences_the_banner_for_that_commit(api_client, local_github):
    repo_id = await _import(api_client, "ack-repo")
    conv = f"conv_ack_{repo_id}"
    await ConversationMemory.get_or_create_conversation(conv, repo_id)
    await ConversationMemory.save_turn(conv, "q", {"answer": "a", "sources": []})

    (local_github / "billing.py").write_text("def charge(a):\n    return a + 5\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "moved", cwd=local_github)
    _branch_cache.clear()

    drift = api_client.get(f"/api/repositories/{repo_id}/drift",
                           params={"conversation_id": conv}).json()
    assert drift["state"] == "behind" and drift["acknowledged"] is False

    response = api_client.post(f"/api/conversations/{conv}/ack-drift",
                               json={"sha": drift["remote_sha"]})
    assert response.status_code == 200

    after = api_client.get(f"/api/repositories/{repo_id}/drift",
                           params={"conversation_id": conv}).json()
    assert after["state"] == "behind", "acknowledging must not pretend the repository is current"
    assert after["acknowledged"] is True, "the banner would reappear for a commit already declined"


async def test_a_further_commit_raises_the_question_again(api_client, local_github):
    """Acknowledging one commit is not silence forever."""
    repo_id = await _import(api_client, "reask-repo")
    conv = f"conv_reask_{repo_id}"
    await ConversationMemory.get_or_create_conversation(conv, repo_id)
    await ConversationMemory.save_turn(conv, "q", {"answer": "a", "sources": []})

    (local_github / "billing.py").write_text("def charge(a):\n    return a + 1\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "first", cwd=local_github)
    _branch_cache.clear()

    first = api_client.get(f"/api/repositories/{repo_id}/drift",
                           params={"conversation_id": conv}).json()
    api_client.post(f"/api/conversations/{conv}/ack-drift", json={"sha": first["remote_sha"]})

    (local_github / "billing.py").write_text("def charge(a):\n    return a + 2\n")
    _git("add", ".", cwd=local_github)
    _git("commit", "-m", "second", cwd=local_github)
    _branch_cache.clear()

    second = api_client.get(f"/api/repositories/{repo_id}/drift",
                            params={"conversation_id": conv}).json()
    assert second["remote_sha"] != first["remote_sha"]
    assert second["acknowledged"] is False, "a new commit was silenced by an old acknowledgement"


def test_acknowledging_an_unknown_conversation_is_404(api_client):
    response = api_client.post("/api/conversations/nope/ack-drift", json={"sha": "a" * 40})
    assert response.status_code == 404


async def test_acknowledging_without_a_sha_is_rejected(api_client, local_github):
    repo_id = await _import(api_client, "nosha-repo")
    conv = f"conv_nosha_{repo_id}"
    await ConversationMemory.get_or_create_conversation(conv, repo_id)

    assert api_client.post(f"/api/conversations/{conv}/ack-drift",
                           json={"sha": "  "}).status_code == 400
