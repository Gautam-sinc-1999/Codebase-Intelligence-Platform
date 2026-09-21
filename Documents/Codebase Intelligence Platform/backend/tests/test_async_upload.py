"""
Zip upload returns a handle rather than a finished index (G-11).

The RETAIL upload took 36 seconds, about 89 % of it embedding, and a browser or an intermediary
proxy abandons a request long before a large repository finishes.

The interesting part is *where the line falls*. The uploaded stream only exists for the duration
of the request, so saving and extracting the archive has to happen inline; indexing is the slow
half and is what became the job. So a corrupt or hostile archive still fails synchronously — the
right answer for a request that was never going to work — while a valid one returns 202.
"""
import io
import time
import asyncio
import zipfile

import pytest

from app.core.database import db_client
from app.core.jobs import indexing_jobs
from app.ingestion.index_store import index_store


def _zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in entries.items():
            archive.writestr(name, body)
    return buffer.getvalue()


async def _settle(repository_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = indexing_jobs.get(repository_id)
        if job is not None and job.is_terminal:
            await asyncio.sleep(0.05)
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"upload job {repository_id} did not settle")


# ------------------------------------------------------------------ the happy path

async def test_upload_returns_before_indexing_finishes(api_client):
    started = time.time()
    response = api_client.post("/api/repositories/upload", files={
        "file": ("proj.zip", _zip({"proj/a.py": "def alpha():\n    return 1\n"}), "application/zip")})
    elapsed = time.time() - started

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "indexing"

    job = await _settle(body["repository_id"])
    assert job.state == "done"

    # Compared against the job's own duration rather than a fixed number of seconds: a wall-clock
    # bound measures machine load, not behaviour.
    assert elapsed < job.elapsed + 1.0


async def test_the_repository_is_listable_while_it_is_still_indexing(api_client):
    """
    The record is written before the job starts, so the upload appears immediately instead of
    vanishing for half a minute.
    """
    response = api_client.post("/api/repositories/upload", files={
        "file": ("visible.zip", _zip({"p/a.py": "def a():\n    return 1\n"}), "application/zip")})
    repo_id = response.json()["repository_id"]

    listed = {r["repository_id"] for r in api_client.get("/api/repositories").json()}
    assert repo_id in listed, "the upload was invisible until indexing finished"

    await _settle(repo_id)


async def test_the_counts_are_filled_in_when_the_job_finishes(api_client):
    response = api_client.post("/api/repositories/upload", files={
        "file": ("counted.zip", _zip({
            "p/a.py": "def a():\n    return 1\n",
            "p/b.py": "def b():\n    return 2\n",
        }), "application/zip")})
    repo_id = response.json()["repository_id"]
    await _settle(repo_id)

    row = next(r for r in api_client.get("/api/repositories").json()
               if r["repository_id"] == repo_id)
    assert row["file_count"] == 2
    assert row["total_lines"] > 0
    assert "python" in row["languages"]


async def test_status_reports_progress_then_ready(api_client):
    response = api_client.post("/api/repositories/upload", files={
        "file": ("poll.zip", _zip({"p/a.py": "def a():\n    return 1\n"}), "application/zip")})
    repo_id = response.json()["repository_id"]

    await _settle(repo_id)
    status = api_client.get(f"/api/repositories/{repo_id}/status").json()
    assert status["status"] == "ready"
    assert status["index_persisted"] is True


async def test_the_uploaded_code_is_actually_queryable(api_client):
    """An upload that produces no usable index is not an upload."""
    response = api_client.post("/api/repositories/upload", files={
        "file": ("q.zip", _zip({
            "p/billing.py": "def charge(amount):\n    return amount * 2\n"}), "application/zip")})
    repo_id = response.json()["repository_id"]
    await _settle(repo_id)

    chunks = index_store.load(repo_id)
    charge = next(c for c in chunks if c["symbol"] == "charge")

    # The path comes from the archive's own layout — the entry was `p/billing.py`, so that is
    # what a citation records and what the viewer asks for.
    snippet = api_client.get(f"/api/repositories/{repo_id}/snippet",
                             params={"file_path": charge["file_path"],
                                     "start_line": charge["start_line"],
                                     "end_line": charge["end_line"]})
    assert snippet.status_code == 200, f"citation path {charge['file_path']!r} did not resolve"
    assert "amount * 2" in snippet.json()["code"]


# ------------------------------------------------------------------ failures stay synchronous

def test_a_corrupt_archive_still_fails_synchronously(api_client):
    """
    The archive is the caller's, and it was never going to work — that answer belongs in the
    response to the request that caused it, not in a status poll a second later.
    """
    response = api_client.post("/api/repositories/upload", files={
        "file": ("broken.zip", b"this is not a zip", "application/zip")})
    assert response.status_code == 400
    assert "not a valid zip" in response.json()["detail"]


def test_a_non_zip_filename_is_still_rejected_immediately(api_client):
    response = api_client.post("/api/repositories/upload", files={
        "file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_a_path_traversal_archive_is_rejected_before_any_job_starts(api_client):
    hostile = _zip({"../../escape.py": "def evil():\n    return 1\n"})
    response = api_client.post("/api/repositories/upload", files={
        "file": ("evil.zip", hostile, "application/zip")})

    assert response.status_code == 400
    assert "Rejected archive" in response.json()["detail"]


async def test_a_rejected_upload_leaves_no_repository_record(api_client):
    before = {r["repository_id"] for r in api_client.get("/api/repositories").json()}

    api_client.post("/api/repositories/upload", files={
        "file": ("bad.zip", b"not a zip at all", "application/zip")})

    after = {r["repository_id"] for r in api_client.get("/api/repositories").json()}
    assert after == before, "a rejected upload left a row behind"


async def test_an_indexing_failure_removes_the_record(api_client, monkeypatch):
    """
    Same reasoning as a failed GitHub import: a row that cannot answer anything is worse than no
    row, because it appears in the sidebar and fails on every use.
    """
    import app.api.repositories as repos_module

    def explode(*args, **kwargs):
        raise RuntimeError("indexing blew up")

    monkeypatch.setattr(repos_module, "index_repository_folder", explode)

    response = api_client.post("/api/repositories/upload", files={
        "file": ("doomed.zip", _zip({"p/a.py": "def a():\n    return 1\n"}), "application/zip")})
    repo_id = response.json()["repository_id"]

    job = await _settle(repo_id)
    assert job.state == "failed"

    await db_client.connect()
    assert await db_client.get_collection("repositories").find_one(
        {"repository_id": repo_id}) is None

    status = api_client.get(f"/api/repositories/{repo_id}/status").json()
    assert status["status"] == "failed"
    assert "blew up" in status["error"]
