"""
Importing a repository from GitHub.

Deliberately a separate module from `repositories.py`. The zip path is untouched by everything
here: `POST /upload` keeps its synchronous contract and its response shape, so the existing
frontend continues to work unchanged. Only the *acquisition* differs — `clone → directory` in
place of `extract → directory` — and both hand the same plain directory to
`index_repository_folder`.

Unlike upload, this path is asynchronous from birth. Cloning and indexing a real repository takes
tens of seconds, and a browser or proxy will abandon the request long before it finishes, so the
import returns `202 Accepted` with an id the client polls.
"""
import os
import shutil
import asyncio
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import db_client
from app.core.jobs import indexing_jobs, JobAlreadyRunning
from app.ingestion.git_source import (
    GitSourceError,
    InvalidRepositoryURL,
    CloneFailed,
    RepositoryTooLarge,
    validate_github_url,
    validate_branch_name,
    derive_repository_id,
    list_remote_branches,
    clone_repository,
    enforce_clone_limits,
    remote_head_sha,
)
from app.ingestion.index_store import index_store
from app.api.repositories import (
    index_repository_folder,
    purge_repository_stores,
    indexing_status,
)
from app.observability import tracing

logger = logging.getLogger("api.github")

# Registered before the repositories router so that `/from-github` and `/github/...` are matched
# before `/{repository_id}` can swallow them.
router = APIRouter(prefix="/repositories", tags=["github"])


class GitHubImportRequest(BaseModel):
    url: str
    branch: Optional[str] = None


def _clone_dir(repository_id: str) -> str:
    """
    Where a clone lives while it is being indexed.

    Under DATA_DIR rather than the system temp directory so that a clone orphaned by a crash is
    found in an obvious place, and so it is removed by the same cleanup that handles everything
    else about a repository.
    """
    return os.path.join(os.path.abspath(settings.DATA_DIR), "clones", repository_id)


def _clone_and_index(repository_id: str, clone_url: str, branch: str, name: str) -> dict:
    """
    The whole synchronous pipeline, run on a worker thread.

    The working tree is deleted as soon as it has been indexed. The chunk index holds everything
    retrieval needs, so keeping a copy of the source would add disk cost and the liability of
    storing someone's code, for no capability. The consequence — `/reindex` cannot work for a git
    repository because there is no `storage_path` — is handled by `/sync`, which re-clones.
    """
    destination = _clone_dir(repository_id)
    with _acquire_trace(repository_id, clone_url, branch, name, "github_import") as trace:
        try:
            commit_sha = _clone_with_limits(clone_url, branch, destination, name)
            result = index_repository_folder(destination, repository_id, name,
                                             job_id=repository_id, trigger="github_import")
            result["commit_sha"] = commit_sha
            trace.update(output={"status": "ok", "commit_sha": commit_sha,
                                 "files": result["file_count"]})
            return result
        except Exception as e:
            trace.update(output={"status": "failed", "error": str(e)})
            raise
        finally:
            # Inside the trace, so the cost and outcome of removing the working tree are visible.
            # This is the step that keeps a failed import from leaving a clone on disk, and it is
            # the one nobody would otherwise notice had stopped happening.
            with tracing.span("cleanup", input={"path": destination}) as s:
                shutil.rmtree(destination, ignore_errors=True)
                s.update(output={"removed": not os.path.isdir(destination)})


@contextmanager
def _acquire_trace(repository_id: str, clone_url: str, branch: str, name: str, trigger: str):
    """
    The root trace for acquiring a repository: clone, size limits, index, cleanup.

    Opened here rather than inherited, for the same reason indexing opens its own — this runs on a
    `run_in_executor` worker thread, which does not carry the request's context across. The index
    trace nests underneath automatically, so one trace covers the whole job.
    """
    with tracing.start_trace_sync(
        "acquire",
        metadata={"repository_id": repository_id, "repository": name, "branch": branch,
                  "clone_url": clone_url, "trigger": trigger, "job_id": repository_id},
        tags=["acquire", trigger],
    ) as trace:
        yield trace


def _clone_with_limits(clone_url: str, branch: str, destination: str, name: str) -> str:
    """Clones and enforces the size limits, tracing each. Returns the commit sha."""
    with tracing.span("clone", input={"clone_url": clone_url, "branch": branch}) as s:
        commit_sha = clone_repository(clone_url, branch, destination)
        s.update(output={"commit_sha": commit_sha})

    with tracing.span("enforce_limits") as s:
        stats = enforce_clone_limits(destination)
        s.update(output={"file_count": stats["file_count"],
                         "total_bytes": stats["total_bytes"]})

    logger.info(
        "Cloned %s@%s at %s (%d files, %.1f MB).",
        name, branch, commit_sha[:12], stats["file_count"], stats["total_bytes"] / 1048576,
    )
    return commit_sha


@router.get("/github/branches")
async def github_branches(url: str):
    """
    Lists a repository's branches without cloning it.

    Backed by `git ls-remote`, which costs about a second and needs no token for a public
    repository. The GitHub REST API would need one to escape a 60/hour anonymous rate limit, so
    for this purpose ls-remote is strictly better.

    Results are cached per URL for a few minutes inside `git_source`: the UI calls this on every
    paste, and branch lists do not change by the second.
    """
    try:
        repo_ref = validate_github_url(url)
    except InvalidRepositoryURL as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = await asyncio.to_thread(list_remote_branches, repo_ref.clone_url)
    except CloneFailed as e:
        # Reaching GitHub failed; that is upstream's problem, not a malformed request.
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach '{repo_ref.name}' on GitHub: {e}",
        )

    if not result.get("branches"):
        raise HTTPException(status_code=422, detail=f"'{repo_ref.name}' has no branches.")

    # Which branches are already indexed, so the picker can say so rather than making the user
    # remember. Each branch is its own repository id by construction.
    repo_col = db_client.get_collection("repositories")
    for branch in result["branches"]:
        existing = await repo_col.find_one(
            {"repository_id": derive_repository_id(repo_ref.owner, repo_ref.repo, branch["name"])}
        )
        branch["indexed"] = existing is not None
        branch["synced_at"] = existing.get("synced_at") if existing else None

    result["owner"] = repo_ref.owner
    result["repo"] = repo_ref.repo
    result["name"] = repo_ref.name
    return result


@router.post("/from-github", status_code=202)
async def import_from_github(request: GitHubImportRequest):
    """
    Starts an import and returns immediately with an id to poll.

    The repository id is **derived** from `owner/repo@branch` rather than freshly generated, so
    importing the same branch twice updates one repository instead of creating two unrelated
    copies of it — which is what a zip upload does, and is wrong for something with an upstream.
    """
    # Traced separately from the import itself: this part happens inside the request, before the
    # 202, while the clone happens on a worker thread afterwards. They are different lifetimes, so
    # they are different traces — and this is the one that shows a rejected paste or a slow GitHub.
    async with tracing.start_trace(
        "resolve_repository",
        metadata={"url": request.url, "requested_branch": request.branch},
        tags=["acquire", "resolve"],
    ):
        with tracing.span("validate_url", input={"url": request.url}) as s:
            try:
                repo_ref = validate_github_url(request.url)
            except InvalidRepositoryURL as e:
                s.update(output={"valid": False, "reason": str(e)})
                raise HTTPException(status_code=400, detail=str(e))
            s.update(output={"valid": True, "owner": repo_ref.owner, "repo": repo_ref.repo})

        # Resolving the default branch costs about a second and makes the 202 response meaningful:
        # the client learns which branch it actually got rather than having to ask afterwards.
        if request.branch:
            with tracing.span("validate_branch", input={"branch": request.branch}) as s:
                try:
                    branch = validate_branch_name(request.branch)
                except InvalidRepositoryURL as e:
                    s.update(output={"valid": False, "reason": str(e)})
                    raise HTTPException(status_code=400, detail=str(e))
                s.update(output={"valid": True, "branch": branch})
        else:
            with tracing.span("ls_remote", input={"clone_url": repo_ref.clone_url}) as s:
                try:
                    remote = await asyncio.to_thread(list_remote_branches, repo_ref.clone_url)
                except CloneFailed as e:
                    s.update(output={"reachable": False, "reason": str(e)})
                    raise HTTPException(
                        status_code=502,
                        detail=f"Could not reach '{repo_ref.name}' on GitHub: {e}",
                    )
                branch = remote.get("default_branch")
                s.update(output={"reachable": True, "default_branch": branch,
                                 "branches": len(remote.get("branches", []))})
            if not branch:
                raise HTTPException(
                    status_code=422,
                    detail=f"'{repo_ref.name}' has no branches to index.",
                )

    repository_id = derive_repository_id(repo_ref.owner, repo_ref.repo, branch)
    name = f"{repo_ref.name}@{branch}"

    if indexing_jobs.is_active(repository_id):
        raise HTTPException(
            status_code=409,
            detail=f"'{name}' is already being imported.",
        )

    repo_col = db_client.get_collection("repositories")
    existing = await repo_col.find_one({"repository_id": repository_id})

    record = {
        "repository_id": repository_id,
        "name": name,
        "source": "github",
        "clone_url": repo_ref.clone_url,
        "owner": repo_ref.owner,
        "repo": repo_ref.repo,
        "branch": branch,
        # No storage_path: the clone is deleted after indexing. /sync re-clones instead.
        "file_count": 0,
        "total_lines": 0,
        "languages": [],
    }
    if existing:
        await repo_col.update_one({"repository_id": repository_id}, {"$set": record})
    else:
        await repo_col.insert_one(dict(record))

    async def on_success(job_id: str, result: dict) -> None:
        await repo_col.update_one(
            {"repository_id": job_id},
            {"$set": {
                "file_count": result["file_count"],
                "total_lines": result["total_lines"],
                "languages": result["languages"],
                "commit_sha": result["commit_sha"],
                "synced_at": datetime.utcnow().isoformat(),
            }},
        )
        logger.info("Imported %s (%d files).", name, result["file_count"])

    async def on_failure(job_id: str, exc: BaseException) -> None:
        # Everything derived goes, and so does the record: a repository row that cannot answer
        # anything is worse than no row, because it shows in the sidebar and fails on every use.
        # The failure itself stays readable through /status, which consults the job runner.
        purge_repository_stores(job_id)
        await repo_col.delete_one({"repository_id": job_id})
        indexing_status[job_id] = f"failed: {exc}"
        logger.error("Import of %s failed: %s", name, exc)

    try:
        indexing_jobs.submit(
            repository_id,
            _clone_and_index,
            repository_id, repo_ref.clone_url, branch, name,
            on_success=on_success,
            on_failure=on_failure,
        )
    except JobAlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e))

    indexing_status[repository_id] = "indexing"
    return {
        "repository_id": repository_id,
        "name": name,
        "owner": repo_ref.owner,
        "repo": repo_ref.repo,
        "branch": branch,
        "status": "indexing",
        "reindexed": bool(existing),
    }


def _clone_and_reindex(repository_id: str, clone_url: str, branch: str,
                       name: str, force_full: bool) -> dict:
    """
    Re-clones and re-indexes. The incremental path is what makes this cheap.

    A fresh clone is a brand-new directory, so nothing about it is recognisable by path or
    timestamp — but the manifest keys on **content hash** (F-23), so every file whose bytes are
    unchanged still matches and keeps its existing chunks. Only genuinely changed files are
    re-parsed and re-embedded, which matters because embedding is ~89 % of indexing time.
    """
    destination = _clone_dir(repository_id)
    with _acquire_trace(repository_id, clone_url, branch, name, "github_sync") as trace:
        try:
            commit_sha = _clone_with_limits(clone_url, branch, destination, name)
            result = index_repository_folder(destination, repository_id, name, force_full,
                                             job_id=repository_id, trigger="github_sync")
            result["commit_sha"] = commit_sha
            # The reuse counts are what make a sync cheap; on the trace they sit next to the
            # clone that produced them, which is where "why was this sync slow?" gets answered.
            trace.update(output={"status": "ok", "commit_sha": commit_sha,
                                 "files": result["file_count"],
                                 "reused_files": result["reused_files"],
                                 "parsed_files": result["parsed_files"]})
            return result
        except Exception as e:
            trace.update(output={"status": "failed", "error": str(e)})
            raise
        finally:
            with tracing.span("cleanup", input={"path": destination}) as s:
                shutil.rmtree(destination, ignore_errors=True)
                s.update(output={"removed": not os.path.isdir(destination)})


@router.post("/{repository_id}/sync")
async def sync_repository(repository_id: str, full: bool = False):
    """
    Brings a GitHub repository up to date with its branch.

    This is the counterpart to `/reindex`, which cannot work here: the clone is deleted after
    indexing, so there is no `storage_path` to re-read. Sync re-acquires the source first, which
    is the whole difference between the two.

    Returns **200** when the branch has not moved and the index is intact — there is nothing to
    do, and saying so costs one `ls-remote` rather than a clone and a re-index. Otherwise
    **202**, with the work running in the background as it does for an import.
    """
    repo_col = db_client.get_collection("repositories")
    repo = await repo_col.find_one({"repository_id": repository_id})
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    if repo.get("source") != "github":
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{repo.get('name', repository_id)}' was not imported from GitHub, so there is "
                f"nothing to sync from. Use POST /repositories/{repository_id}/reindex instead."
            ),
        )

    clone_url = repo.get("clone_url")
    branch = repo.get("branch")
    if not clone_url or not branch:
        raise HTTPException(
            status_code=409,
            detail="This repository is missing its upstream details; re-import it from its URL.",
        )

    if indexing_jobs.is_active(repository_id):
        raise HTTPException(status_code=409, detail="This repository is already being indexed.")

    name = repo.get("name", repository_id)
    current_sha = repo.get("commit_sha")

    # Deliberately uncached: a stale answer here would report a repository as current when it is
    # not, which is the one thing this check exists to prevent.
    try:
        upstream_sha = await asyncio.to_thread(remote_head_sha, clone_url, branch)
    except GitSourceError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach GitHub: {e}")

    if upstream_sha is None:
        raise HTTPException(
            status_code=410,
            detail=f"Branch '{branch}' no longer exists on the remote.",
        )

    already_current = (
        not full
        and current_sha == upstream_sha
        and index_store.exists(repository_id)
    )
    if already_current:
        return {
            "repository_id": repository_id,
            "name": name,
            "branch": branch,
            "status": "already current",
            "commit_sha": current_sha,
            "synced_at": repo.get("synced_at"),
        }

    async def on_success(job_id: str, result: dict) -> None:
        await repo_col.update_one(
            {"repository_id": job_id},
            {"$set": {
                "file_count": result["file_count"],
                "total_lines": result["total_lines"],
                "languages": result["languages"],
                "commit_sha": result["commit_sha"],
                "synced_at": datetime.utcnow().isoformat(),
            }},
        )
        # Every thread about this repository now cites code that may have moved. Recording the
        # boundary in the transcript is what makes the stale labels on old citations read as an
        # honest statement rather than a malfunction.
        from app.memory.conversation_memory import ConversationMemory
        marked = await ConversationMemory.mark_repository_synced(
            job_id, current_sha or "", result["commit_sha"], result["parsed_files"]
        )
        logger.info(
            "Synced %s to %s: %d file(s) reused, %d re-parsed, %d thread(s) marked.",
            name, result["commit_sha"][:12], result["reused_files"], result["parsed_files"], marked,
        )

    async def on_failure(job_id: str, exc: BaseException) -> None:
        # Unlike a failed import, the record stays. The repository already existed and its
        # previous index is still on disk and still answerable; a failed sync should leave the
        # user where they were, not destroy a working repository because a network call failed.
        indexing_status[job_id] = f"failed: {exc}"
        logger.error("Sync of %s failed: %s", name, exc)

    try:
        indexing_jobs.submit(
            repository_id,
            _clone_and_reindex,
            repository_id, clone_url, branch, name, full,
            on_success=on_success,
            on_failure=on_failure,
        )
    except JobAlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e))

    indexing_status[repository_id] = "indexing"
    return JSONResponse(
        status_code=202,
        content={
            "repository_id": repository_id,
            "name": name,
            "branch": branch,
            "status": "syncing",
            "mode": "full" if full else "incremental",
            "from_commit": current_sha,
            "to_commit": upstream_sha,
        },
    )


@router.get("/{repository_id}/drift")
async def repository_drift(repository_id: str, conversation_id: Optional[str] = None):
    """
    Reports whether a repository has fallen behind its branch.

    Called when a thread is reopened. **Never blocks the thread from opening**: messages render
    from local disk in milliseconds while this costs a network round trip, so the client fires it
    afterwards and shows the result when it arrives.

    Three outcomes, not two. The obvious comparison is the index against the remote, but as soon
    as one repository has more than one thread a third case appears:

    - `current`         — nothing to do
    - `behind`          — upstream moved; offer a sync
    - `thread_behind`   — *another thread already synced this repository*, so this thread's
                          answers describe a version that is no longer indexed. Nothing upstream
                          changed and there is nothing to pull, so offering a sync here would be
                          a button that does nothing.

    A repository with no upstream (a zip) is never behind; it is reported as current without any
    network call at all.
    """
    repo_col = db_client.get_collection("repositories")
    repo = await repo_col.find_one({"repository_id": repository_id})
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    indexed_sha = repo.get("commit_sha") or ""
    thread_sha = ""
    acknowledged = ""

    if conversation_id:
        from app.memory.mongo_thread_store import thread_store
        meta = await thread_store.read_meta(conversation_id) or {}
        thread_sha = meta.get("indexed_commit_sha") or ""
        acknowledged = meta.get("drift_ack_sha") or ""

    base = {
        "repository_id": repository_id,
        "name": repo.get("name", repository_id),
        "source": repo.get("source", "zip"),
        "branch": repo.get("branch"),
        "indexed_sha": indexed_sha,
        "thread_sha": thread_sha,
        "checked_at": datetime.utcnow().isoformat(),
    }

    if repo.get("source") != "github" or not repo.get("clone_url") or not repo.get("branch"):
        # No upstream to be behind of. Answered without touching the network.
        return {**base, "state": "current", "behind": False, "remote_sha": None}

    # This thread is behind the index itself — someone else synced. Reported before consulting
    # the remote, because it is true regardless of what the remote says and needs no network call.
    if thread_sha and indexed_sha and thread_sha != indexed_sha:
        return {
            **base,
            "state": "thread_behind",
            "behind": False,
            "remote_sha": None,
            "detail": (
                "This repository has been re-indexed since this thread's last answer. Earlier "
                "answers describe an older version of the code."
            ),
        }

    try:
        # Cached briefly: this runs on every thread open, and noticing a commit a few minutes
        # late is a freshness question rather than a correctness one. /sync stays uncached.
        remote_sha = await asyncio.to_thread(
            remote_head_sha, repo["clone_url"], repo["branch"], use_cache=True
        )
    except GitSourceError as e:
        # Unreachable upstream must never stop a thread being used.
        return {**base, "state": "unknown", "behind": False, "remote_sha": None, "error": str(e)}

    if remote_sha is None:
        return {
            **base,
            "state": "branch_gone",
            "behind": False,
            "remote_sha": None,
            "detail": f"Branch '{repo['branch']}' no longer exists on the remote.",
        }

    behind = bool(indexed_sha) and remote_sha != indexed_sha
    return {
        **base,
        "state": "behind" if behind else "current",
        "behind": behind,
        "remote_sha": remote_sha,
        # The client suppresses the banner for a SHA the user chose to stay on, so it appears
        # once per new commit rather than on every message.
        "acknowledged": bool(acknowledged) and acknowledged == remote_sha,
    }
