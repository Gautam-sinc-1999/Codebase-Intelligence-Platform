"""
The GitHub import endpoint.

Network access is confined to `clone_repository`, which is patched to clone from a local git
repository built in `tmp_path`. Everything else — validation, derived identity, the 202 contract,
job supervision, record writing, failure cleanup — is the real code path.

The other standing requirement is that none of this disturbs zip upload, which has its own test
at the end.
"""
import asyncio
import os
import subprocess
import time

import pytest

import app.api.github as github_module
from app.core.database import db_client
from app.core.jobs import indexing_jobs
from app.ingestion.git_source import clone_repository, derive_repository_id
from app.ingestion.index_store import index_store


def _git(*args, cwd):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    return subprocess.run(["git", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True)


@pytest.fixture
def origin_repo(tmp_path):
    """A real repository with two branches and cross-file calls worth indexing."""
    root = tmp_path / "origin"
    root.mkdir()
    _git("init", "-b", "main", cwd=root)
    (root / "api.py").write_text(
        "def handler(request):\n"
        "    return calculate_total(request.items)\n"
        "\n"
        "def calculate_total(items):\n"
        "    return sum(i.price for i in items)\n"
    )
    _git("add", ".", cwd=root)
    _git("commit", "-m", "initial", cwd=root)

    _git("checkout", "-b", "develop", cwd=root)
    (root / "extra.py").write_text("def extra():\n    return 1\n")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "develop", cwd=root)
    _git("checkout", "main", cwd=root)
    return root


@pytest.fixture
def local_github(monkeypatch, origin_repo):
    """
    Redirects cloning and branch listing at the local repository.

    Patched at the seam rather than deeper: the endpoint's validation, id derivation and job
    handling all run for real, and only the bytes come from disk instead of github.com.
    """
    def fake_clone(url, branch, dest, **kwargs):
        return clone_repository(str(origin_repo), branch, dest, allow_local=True)

    def fake_branches(url, **kwargs):
        return {
            "default_branch": "main",
            "branches": [{"name": "main", "commit_sha": "a" * 40, "is_default": True},
                         {"name": "develop", "commit_sha": "b" * 40, "is_default": False}],
        }

    monkeypatch.setattr(github_module, "clone_repository", fake_clone)
    monkeypatch.setattr(github_module, "list_remote_branches", fake_branches)
    return origin_repo


async def _await_import(api_client, repository_id, timeout=30):
    """Polls /status the way the frontend will, rather than reaching into the runner."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = indexing_jobs.get(repository_id)
        if job is not None and job.is_terminal:
            await asyncio.sleep(0.05)   # let the completion callback run on the loop
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"import of {repository_id} did not finish within {timeout}s")


# ---------------------------------------------------------------- the happy path

async def test_import_returns_202_immediately_then_indexes(api_client, local_github):
    started = time.time()
    response = api_client.post("/api/repositories/from-github",
                               json={"url": "https://github.com/acme/widgets"})
    elapsed = time.time() - started

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "indexing"
    assert body["branch"] == "main"
    assert body["name"] == "acme/widgets@main"

    job = await _await_import(api_client, body["repository_id"])
    assert job.state == "done"

    # The request must have returned *before* the work finished — the definition of handing back
    # a handle rather than a result.
    #
    # Compared against the job's own duration rather than a fixed number of seconds: an absolute
    # wall-clock bound is a function of machine load, and failed intermittently under a full
    # suite run while passing every time in isolation. Both figures move together with load, so
    # the ratio tests the invariant and nothing else.
    assert elapsed < job.elapsed, (
        f"request took {elapsed:.2f}s against a {job.elapsed:.2f}s job — it waited for indexing"
    )

    status = api_client.get(f"/api/repositories/{body['repository_id']}/status").json()
    assert status["status"] == "ready"
    assert status["source"] == "github"
    assert status["branch"] == "main"
    assert len(status["commit_sha"]) == 40


async def test_imported_code_is_actually_queryable(api_client, local_github):
    """An import that produces no usable index is not an import."""
    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets"}).json()
    await _await_import(api_client, body["repository_id"])

    repo_id = body["repository_id"]
    assert index_store.exists(repo_id)

    chunks = index_store.load(repo_id)
    symbols = {c["symbol"] for c in chunks}
    assert "calculate_total" in symbols and "handler" in symbols

    snippet = api_client.get(f"/api/repositories/{repo_id}/snippet",
                             params={"file_path": "api.py", "start_line": 4, "end_line": 5})
    assert snippet.status_code == 200
    assert "sum(i.price" in snippet.json()["code"]


async def test_repository_record_carries_its_upstream_identity(api_client, local_github):
    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets"}).json()
    await _await_import(api_client, body["repository_id"])

    await db_client.connect()
    record = await db_client.get_collection("repositories").find_one(
        {"repository_id": body["repository_id"]})

    assert record["source"] == "github"
    assert record["owner"] == "acme" and record["repo"] == "widgets"
    assert record["branch"] == "main"
    assert record["clone_url"] == "https://github.com/acme/widgets.git"
    assert record["file_count"] == 1 and record["total_lines"] > 0
    assert record.get("storage_path") is None, "the clone should not be retained"
    assert record["synced_at"] and len(record["commit_sha"]) == 40


async def test_the_working_tree_is_deleted_after_indexing(api_client, local_github):
    """Not storing someone's source removes a liability and the index needs nothing more."""
    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets"}).json()
    await _await_import(api_client, body["repository_id"])

    assert not os.path.exists(github_module._clone_dir(body["repository_id"]))


async def test_explicit_branch_is_honoured(api_client, local_github):
    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets", "branch": "develop"}).json()
    assert body["branch"] == "develop"
    await _await_import(api_client, body["repository_id"])

    symbols = {c["symbol"] for c in index_store.load(body["repository_id"])}
    assert "extra" in symbols, "indexed the wrong branch"


# ---------------------------------------------------------------- identity

async def test_importing_the_same_branch_twice_yields_one_repository(api_client, local_github):
    """
    A zip upload mints a fresh uuid each time, so the same project uploaded twice becomes two
    unrelated repositories. For something with an upstream that is wrong: re-importing must
    update what is already indexed.
    """
    first = api_client.post("/api/repositories/from-github",
                            json={"url": "https://github.com/acme/widgets"}).json()
    await _await_import(api_client, first["repository_id"])

    second = api_client.post("/api/repositories/from-github",
                             json={"url": "https://github.com/acme/widgets"}).json()
    await _await_import(api_client, second["repository_id"])

    assert first["repository_id"] == second["repository_id"]
    assert second["reindexed"] is True

    await db_client.connect()
    rows = await db_client.get_collection("repositories").find(
        {"repository_id": first["repository_id"]}).to_list(None)
    assert len(rows) == 1, f"re-import created a duplicate row ({len(rows)} found)"


async def test_two_branches_are_two_repositories_with_isolated_indexes(api_client, local_github):
    """The F-45 regression, now reachable by ordinary use rather than only by a test."""
    main = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets", "branch": "main"}).json()
    await _await_import(api_client, main["repository_id"])

    dev = api_client.post("/api/repositories/from-github",
                          json={"url": "https://github.com/acme/widgets", "branch": "develop"}).json()
    await _await_import(api_client, dev["repository_id"])

    assert main["repository_id"] != dev["repository_id"]

    main_symbols = {c["symbol"] for c in index_store.load(main["repository_id"])}
    dev_symbols = {c["symbol"] for c in index_store.load(dev["repository_id"])}
    assert "extra" in dev_symbols
    assert "extra" not in main_symbols, "one branch's symbols leaked into the other"


async def test_derived_id_matches_the_module_that_computes_it(api_client, local_github):
    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets"}).json()
    assert body["repository_id"] == derive_repository_id("acme", "widgets", "main")
    await _await_import(api_client, body["repository_id"])


# ---------------------------------------------------------------- rejection and failure

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "https://github.com@evil.com/a/b",
    "https://github.com.evil.com/a/b",
    "--upload-pack=touch /tmp/pwned",
    "not a url at all",
])
def test_hostile_urls_are_refused_with_400(api_client, url):
    response = api_client.post("/api/repositories/from-github", json={"url": url})
    assert response.status_code == 400


def test_a_malformed_branch_is_refused_with_400(api_client):
    response = api_client.post(
        "/api/repositories/from-github",
        json={"url": "https://github.com/acme/widgets", "branch": "--upload-pack=sh"})
    assert response.status_code == 400


async def test_a_failed_clone_leaves_nothing_behind(api_client, monkeypatch, local_github):
    """
    A network clone fails far more often than a zip upload, so the cleanup path is the normal
    path here, not an edge case.
    """
    from app.ingestion.git_source import CloneFailed

    def failing_clone(url, branch, dest, **kwargs):
        raise CloneFailed("Repository not found.")

    monkeypatch.setattr(github_module, "clone_repository", failing_clone)

    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/gone"}).json()
    repo_id = body["repository_id"]
    job = await _await_import(api_client, repo_id)

    assert job.state == "failed"

    await db_client.connect()
    assert await db_client.get_collection("repositories").find_one(
        {"repository_id": repo_id}) is None, "a broken repository row survived"
    assert not index_store.exists(repo_id)
    assert not os.path.exists(github_module._clone_dir(repo_id))

    # The reason is still retrievable even though the record is gone.
    status = api_client.get(f"/api/repositories/{repo_id}/status").json()
    assert status["status"] == "failed"
    assert "not found" in status["error"].lower()


async def test_a_second_import_while_one_is_running_is_refused(api_client, monkeypatch, local_github):
    """Two imports of one repository would race each other through the same stores."""
    import threading
    release = threading.Event()

    def slow_clone(url, branch, dest, **kwargs):
        release.wait(timeout=10)
        return clone_repository(str(local_github), branch, dest, allow_local=True)

    monkeypatch.setattr(github_module, "clone_repository", slow_clone)

    first = api_client.post("/api/repositories/from-github",
                            json={"url": "https://github.com/acme/widgets"})
    assert first.status_code == 202

    second = api_client.post("/api/repositories/from-github",
                             json={"url": "https://github.com/acme/widgets"})
    assert second.status_code == 409

    release.set()
    await _await_import(api_client, first.json()["repository_id"])


async def test_an_oversized_repository_is_rejected_and_cleaned_up(
        api_client, monkeypatch, local_github):
    from app.core.config import settings

    monkeypatch.setattr(settings, "MAX_ARCHIVE_ENTRIES", 0)

    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/widgets"}).json()
    job = await _await_import(api_client, body["repository_id"])

    assert job.state == "failed"
    assert not os.path.exists(github_module._clone_dir(body["repository_id"]))


# ---------------------------------------------------------------- the zip path is untouched

async def test_zip_upload_accepts_and_indexes_in_the_background(api_client):
    """
    Zip upload became asynchronous in its own right (G-11), after the GitHub work was complete.

    Through increments 4-9 this test asserted the opposite — that upload was still synchronous —
    and that was the point: the zip path was deliberately left untouched so the GitHub feature
    carried no risk to it. The contract changed on purpose afterwards, so the test moved with it.
    """
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("proj/app.py", "def alpha():\n    return 1\n")
    buffer.seek(0)

    response = api_client.post(
        "/api/repositories/upload",
        files={"file": ("proj.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "indexing"
    assert body["source"] == "zip"
    assert body["repository_id"].startswith("repo_"), "zip ids should stay uuid-based"

    job = await _await_import(api_client, body["repository_id"])
    assert job.state == "done"

    status = api_client.get(f"/api/repositories/{body['repository_id']}/status").json()
    assert status["status"] == "ready"


def test_the_literal_github_routes_are_not_captured_by_the_id_route(api_client):
    """
    `/repositories/{repository_id}` would happily match 'from-github' if the routers were
    registered the other way round.
    """
    response = api_client.post("/api/repositories/from-github", json={"url": "bad"})
    assert response.status_code == 400, "expected URL validation, not a repository lookup"


# ---------------------------------------------------------------- branch listing

async def test_branches_are_listed_with_the_default_first(api_client, local_github):
    response = api_client.get("/api/repositories/github/branches",
                              params={"url": "https://github.com/acme/widgets"})
    assert response.status_code == 200
    body = response.json()

    assert body["default_branch"] == "main"
    assert body["owner"] == "acme" and body["repo"] == "widgets"
    assert [b["name"] for b in body["branches"]] == ["main", "develop"]
    assert body["branches"][0]["is_default"] is True


async def test_branch_listing_marks_which_branches_are_already_indexed(api_client, local_github):
    """
    The picker should say so rather than making the user remember what they imported.

    Uses an owner/repo no other test touches. Repository ids are *derived* from owner/repo@branch
    by design, so two tests naming the same repository share a row in the session's database —
    which is the identity feature working, not leakage to be reset away.
    """
    url = "https://github.com/acme/never-imported-elsewhere"

    before = api_client.get("/api/repositories/github/branches", params={"url": url}).json()
    assert all(b["indexed"] is False for b in before["branches"])

    body = api_client.post("/api/repositories/from-github",
                           json={"url": url, "branch": "main"}).json()
    await _await_import(api_client, body["repository_id"])

    after = api_client.get("/api/repositories/github/branches", params={"url": url}).json()
    by_name = {b["name"]: b for b in after["branches"]}
    assert by_name["main"]["indexed"] is True and by_name["main"]["synced_at"]
    assert by_name["develop"]["indexed"] is False


def test_branch_listing_rejects_a_hostile_url(api_client):
    response = api_client.get("/api/repositories/github/branches",
                              params={"url": "file:///etc/passwd"})
    assert response.status_code == 400


def test_branch_listing_is_not_captured_by_the_repository_id_route(api_client):
    """
    `GET /repositories/{repository_id}` would match 'github' if the routers were registered the
    other way round, and the failure would look like a missing repository rather than a routing
    bug.
    """
    response = api_client.get("/api/repositories/github/branches", params={"url": "bad"})
    assert response.status_code == 400, "fell through to the repository lookup"


async def test_repository_listing_reports_which_repositories_have_an_upstream(
        api_client, local_github):
    """
    The sidebar decides between Sync and Re-index from this field, and getting it wrong shows the
    user a 409 telling them to re-upload something they never uploaded.
    """
    body = api_client.post("/api/repositories/from-github",
                           json={"url": "https://github.com/acme/listed", "branch": "main"}).json()
    await _await_import(api_client, body["repository_id"])

    listed = api_client.get("/api/repositories").json()
    imported = next(r for r in listed if r["repository_id"] == body["repository_id"])
    assert imported["source"] == "github"
    assert imported["branch"] == "main" and imported["commit_sha"]


async def test_a_zip_repository_still_reports_itself_as_zip(api_client):
    """Default rather than absent, so no client has to special-case the field's absence."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("proj/app.py", "def alpha():\n    return 1\n")
    buffer.seek(0)

    uploaded = api_client.post(
        "/api/repositories/upload",
        files={"file": ("proj.zip", buffer.getvalue(), "application/zip")},
    ).json()
    assert uploaded["source"] == "zip"
    await _await_import(api_client, uploaded["repository_id"])

    listed = api_client.get("/api/repositories").json()
    row = next(r for r in listed if r["repository_id"] == uploaded["repository_id"])
    assert row["source"] == "zip"
    assert row["branch"] is None
    assert row["file_count"] == 1, "the counts were not filled in when the job finished"
