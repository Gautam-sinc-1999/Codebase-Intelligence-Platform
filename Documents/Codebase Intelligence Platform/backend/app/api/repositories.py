import os
import re
import stat
import time
import asyncio
import logging
import zipfile
import shutil
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.core.database import db_client, sanitize_mongo_doc
from app.ingestion.discovery import FileDiscovery, IGNORE_DIRS
from app.ingestion.ast_parser import MultiLanguageASTParser
from app.ingestion.chunker import HierarchicalCodeChunker
from app.vector.chromadb_client import chroma_store
from app.graph.neo4j_client import neo4j_client
from app.ingestion.index_store import index_store
from app.retrieval.hybrid_retriever import BM25IndexCache
from app.observability import tracing

logger = logging.getLogger("api.repositories")

router = APIRouter(prefix="/repositories", tags=["repositories"])

# In-process cache of repository chunks. This is now a cache in front of the on-disk index
# (see index_store), not the only copy — so a restart or a second worker reloads rather than
# re-parsing, and a miss is recoverable instead of fatal.
repository_chunks_cache: Dict[str, List[Dict[str, Any]]] = {}

# repository_id -> "indexing" | "ready" | "failed: <reason>"
indexing_status: Dict[str, str] = {}


def purge_repository_stores(repository_id: str, source_path: str = None) -> Dict[str, Any]:
    """
    Removes every trace of a repository from all six places it exists.

    A repository is not one object; it is the same identity scattered across a Mongo record, a
    Chroma collection, a Neo4j subgraph, a chunk index and manifest on disk, an extracted source
    tree, and three in-process caches. Cleaning up a subset is what produces the orphans this
    function exists to prevent — embeddings whose repository no longer exists, graph nodes nothing
    can reach, a manifest that makes a rebuilt repository look already-indexed.

    Deliberately best-effort per store: a Neo4j outage must not leave the Chroma collection
    undeleted. Each step reports its own outcome rather than aborting the rest.

    Used by DELETE, and by the indexing failure path — indexing writes embeddings before it builds
    the graph, so a failure in between used to strand a populated collection with no owner.
    """
    outcome: Dict[str, Any] = {"repository_id": repository_id}

    try:
        outcome["vectors_deleted"] = chroma_store.delete_collection(repository_id)
    except Exception as e:
        outcome["vectors_deleted"] = False
        logger.error("Vector cleanup failed for '%s': %s", repository_id, e)

    try:
        outcome["graph_nodes_deleted"] = neo4j_client.delete_repository(repository_id)
    except Exception as e:
        outcome["graph_nodes_deleted"] = 0
        logger.error("Graph cleanup failed for '%s': %s", repository_id, e)

    try:
        index_store.delete(repository_id)
        outcome["index_deleted"] = True
    except Exception as e:
        outcome["index_deleted"] = False
        logger.error("Index cleanup failed for '%s': %s", repository_id, e)

    # The extracted source lives one level above the 'codebase' directory recorded on the record.
    outcome["source_deleted"] = False
    if source_path:
        repos_root = os.path.abspath(os.path.join(settings.DATA_DIR, "repositories"))
        candidate = os.path.abspath(source_path)
        if os.path.basename(candidate) == "codebase":
            candidate = os.path.dirname(candidate)
        # Never delete outside the managed storage root: `storage_path` for a test fixture or a
        # git clone can point anywhere, and this function must not be a path-deletion primitive.
        if candidate.startswith(repos_root + os.sep) and os.path.isdir(candidate):
            shutil.rmtree(candidate, ignore_errors=True)
            outcome["source_deleted"] = not os.path.isdir(candidate)

    repository_chunks_cache.pop(repository_id, None)
    indexing_status.pop(repository_id, None)
    BM25IndexCache.invalidate(repository_id)
    outcome["caches_cleared"] = True

    return outcome


async def reconcile_repositories() -> Dict[str, Any]:
    """
    Drops repository records that can no longer become usable, at startup.

    `DELETE` cleans up properly, but nothing reconciled what was already there. A record whose
    index file *and* source directory have both vanished — a manually cleared `data/`, a restored
    database, a half-finished migration — still appeared in the sidebar as a selectable
    repository and failed on first use.

    A record is kept whenever there is any route back to a working index:

    - its chunk index is on disk, or
    - its extracted source is on disk, so it can be re-indexed, or
    - it is a GitHub import with a clone URL, so `/sync` can re-acquire it. Such a record has no
      `storage_path` **by design** — the clone is deleted after indexing — so treating a missing
      directory as fatal would delete exactly the repositories that are easiest to restore.

    Only a record with none of those is removed, along with any derived state it left behind.
    """
    outcome = {"checked": 0, "removed": [], "kept": 0}

    try:
        repo_col = db_client.get_collection("repositories")
        records = await repo_col.find({}).to_list(length=1000)
    except Exception as e:
        logger.warning("Could not reconcile repositories: %s", e)
        return outcome

    for record in records:
        repository_id = record.get("repository_id")
        if not repository_id:
            continue
        outcome["checked"] += 1

        has_index = index_store.exists(repository_id)
        source_path = record.get("storage_path")
        has_source = bool(source_path) and os.path.exists(source_path)
        recoverable = record.get("source") == "github" and bool(record.get("clone_url"))

        if has_index or has_source or recoverable:
            outcome["kept"] += 1
            continue

        logger.warning(
            "Removing repository '%s' (%s): no index, no source on disk, and no upstream to "
            "re-acquire it from.", repository_id, record.get("name", repository_id),
        )
        purge_repository_stores(repository_id, source_path)
        try:
            await repo_col.delete_one({"repository_id": repository_id})
        except Exception as e:
            logger.error("Could not delete stale record '%s': %s", repository_id, e)
            continue
        outcome["removed"].append(repository_id)

    return outcome


def get_repository_chunks(repository_id: str) -> List[Dict[str, Any]]:
    """
    Returns a repository's chunks, loading the persisted index on first use.

    Callers previously read `repository_chunks_cache` directly, so anything not indexed by *this*
    process looked like an empty repository.
    """
    cached = repository_chunks_cache.get(repository_id)
    if cached:
        return cached

    chunks = index_store.load(repository_id)
    if not chunks:
        return []

    repository_chunks_cache[repository_id] = chunks

    # The vector collection is already persistent, but the graph lives in memory, so it has to
    # be rebuilt for this process. That is cheap next to re-parsing the source.
    neo4j_client.build_repository_graph(repository_id, chunks)
    BM25IndexCache.invalidate(repository_id)
    indexing_status[repository_id] = "ready"
    logger.info("Loaded persisted index for '%s' (%d chunks).", repository_id, len(chunks))
    return chunks


class UnsafeArchiveError(Exception):
    """Raised when an uploaded archive violates a safety limit or escapes its extraction root."""


def _stream_upload_to_disk(upload_file, dest_path: str, max_bytes: int) -> int:
    """
    Copies an upload to disk in chunks, aborting as soon as max_bytes is exceeded.
    Avoids buffering a hostile upload in memory or filling the disk.
    """
    written = 0
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise UnsafeArchiveError(
                    f"Archive exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB"
                )
            buffer.write(chunk)
    return written


def _is_ignorable_archive_entry(name: str) -> bool:
    """
    True for archive entries that indexing would skip anyway.

    Real uploads are dominated by material we never look at: a repository zipped with its
    virtualenv, node_modules, .git and macOS __MACOSX forks. In one measured case 22,827 of
    22,841 entries were ignorable and only 14 were indexable — so extracting everything both
    wasted disk and pushed a legitimate upload past the zip-bomb entry limit.

    Secret files are skipped here too, not merely left unindexed: an uploaded archive routinely
    carries .env files and private keys, and the safest handling is never to write them to disk.
    """
    parts = [p for p in re.split(r"[/\\]", name) if p and p not in (".",)]
    if not parts:
        return True

    if any(part in IGNORE_DIRS for part in parts[:-1]):
        return True

    return FileDiscovery.is_secret_file(parts[-1])


def _safe_extract_zip(zip_path: str, extract_dir: str) -> int:
    """
    Extracts a zip archive, rejecting path traversal ("Zip Slip"), symlinks, and zip bombs.

    Every member is resolved against extract_dir and refused if it lands outside, which blocks
    both absolute paths and '../' sequences. Declared sizes are checked before extraction and
    actual bytes are counted during it, so a lying header cannot slip past.
    """
    extract_root = os.path.realpath(extract_dir)
    os.makedirs(extract_root, exist_ok=True)

    total_uncompressed = 0

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Filter before counting: the limit exists to stop zip bombs, not to reject a repository
        # that happens to have been zipped with its virtualenv.
        members = [m for m in zip_ref.infolist() if not _is_ignorable_archive_entry(m.filename)]
        skipped = len(zip_ref.infolist()) - len(members)
        if skipped:
            logger.info("Skipping %d archive entries in ignored directories or secret files.", skipped)

        if len(members) > settings.MAX_ARCHIVE_ENTRIES:
            raise UnsafeArchiveError(
                f"Archive contains {len(members)} indexable entries, exceeding the limit of "
                f"{settings.MAX_ARCHIVE_ENTRIES}"
            )

        for member in members:
            # Reject symlinks outright — the high 16 bits of external_attr hold the Unix mode.
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise UnsafeArchiveError(f"Archive contains a symlink entry: {member.filename}")

            if member.is_dir():
                continue

            if member.file_size > settings.MAX_ENTRY_BYTES:
                raise UnsafeArchiveError(
                    f"Entry '{member.filename}' is {member.file_size} bytes, "
                    f"exceeding the per-file limit of {settings.MAX_ENTRY_BYTES}"
                )

            total_uncompressed += member.file_size
            if total_uncompressed > settings.MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise UnsafeArchiveError(
                    "Archive exceeds the maximum total uncompressed size of "
                    f"{settings.MAX_TOTAL_UNCOMPRESSED_BYTES // (1024 * 1024)} MB"
                )

            # Path traversal check: the resolved destination must stay inside extract_root.
            target_path = os.path.realpath(os.path.join(extract_root, member.filename))
            if target_path != extract_root and not target_path.startswith(extract_root + os.sep):
                raise UnsafeArchiveError(f"Archive entry escapes the extraction directory: {member.filename}")

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Copy with a hard byte cap so a falsified file_size header cannot overrun the limit.
            written = 0
            with zip_ref.open(member, "r") as src, open(target_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > settings.MAX_ENTRY_BYTES:
                        raise UnsafeArchiveError(
                            f"Entry '{member.filename}' expands beyond its declared size"
                        )
                    dst.write(chunk)

    return total_uncompressed

class RepositoryResponse(BaseModel):
    repository_id: str
    name: str
    file_count: int
    total_lines: int
    languages: List[str]
    # Upstream identity, present only for repositories imported from GitHub. Optional with
    # defaults so the zip path's response is byte-identical to what it was, and so any client
    # written against the old shape keeps working.
    source: str = "zip"
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    synced_at: Optional[str] = None

def index_repository_folder(repo_dir: str, repo_id: str, repo_name: str, force_full: bool = False,
                            *, job_id: Optional[str] = None, trigger: str = "upload") -> Dict[str, Any]:
    """
    Discovers, parses, embeds and graphs a repository folder, then persists the index.

    Synchronous and CPU-bound: callers on the event loop must dispatch it via asyncio.to_thread.

    Traced as its own root, not as a child of whatever was running when it was submitted. Indexing
    is handed to a worker thread by `JobRunner` through `run_in_executor`, which — unlike
    `asyncio.to_thread` — does not copy the caller's context, so an ambient trace would not reach
    here. `job_id` and `trigger` are carried as metadata instead, which is what ties a trace back
    to the job a caller is polling.

    Nobody watches this live, and that is exactly why it is worth tracing: it is where ~89% of the
    compute goes, it is where incremental re-index either works or silently does not, and a failure
    here only surfaces much later as a 409.
    """
    indexing_status[repo_id] = "indexing"
    with tracing.start_trace_sync(
        "index",
        metadata={"repository_id": repo_id, "repository": repo_name, "job_id": job_id,
                  "trigger": trigger, "force_full": force_full},
        tags=["index", trigger],
    ) as trace:
        try:
            result = _run_indexing_pipeline(repo_dir, repo_id, repo_name, force_full, trace)
        except Exception as e:
            indexing_status[repo_id] = f"failed: {e}"
            logger.error("Indexing failed for '%s': %s", repo_id, e)
            trace.update(output={"status": "failed", "error": str(e)})
            raise

        indexing_status[repo_id] = "ready"
        trace.update(output={
            "status": "ready",
            "files": result["file_count"],
            "chunks": len(result["all_chunks"]),
            "reused_files": result["reused_files"],
            "parsed_files": result["parsed_files"],
        })
        return result


def _run_indexing_pipeline(repo_dir: str, repo_id: str, repo_name: str, force_full: bool,
                           trace) -> Dict[str, Any]:
    """
    The six stages, each its own span.

    Discovery, parsing and chunking share one span because they share one streaming loop: files are
    parsed as they are discovered so that only the current file is held in memory (F-24). Splitting
    them into three spans would mean three passes and would undo that. Their costs are separated by
    accumulating the time each takes and reporting it on the one span instead.
    """
    with tracing.span("load_previous_index", input={"force_full": force_full}) as s:
        # Incremental re-index. FileDiscovery has always computed a SHA-256 per file; nothing
        # read it, so every re-index re-parsed and re-embedded the entire repository even when
        # a single file had changed. Chunks for files whose hash is unchanged are reused from
        # the stored index, and only changed or new files are parsed again.
        previous_manifest = {} if force_full else index_store.load_manifest(repo_id)
        previous_chunks = [] if force_full else (index_store.load(repo_id) or [])

        chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in previous_chunks:
            chunks_by_file.setdefault(chunk.get("file_path", ""), []).append(chunk)
        s.update(output={"known_files": len(previous_manifest),
                         "previous_chunks": len(previous_chunks),
                         "incremental": bool(previous_chunks)})

    all_chunks: List[Dict[str, Any]] = []
    changed_chunks: List[Dict[str, Any]] = []
    manifest: Dict[str, str] = {}
    languages = set()
    reused_files = 0
    parsed_files = 0
    file_count = 0
    total_lines = 0
    parse_seconds = 0.0
    chunk_seconds = 0.0

    with tracing.span("discover_parse_chunk", input={"path": repo_dir}) as s:
        # Streamed, so only the file being parsed is held in memory (F-24).
        discovered_files = FileDiscovery.iter_repository(repo_dir, repo_id)

        for f_meta in discovered_files:
            file_count += 1
            total_lines += f_meta["line_count"]
            languages.add(f_meta["language"])
            rel_path = f_meta["relative_path"]
            manifest[rel_path] = f_meta["file_hash"]

            unchanged = (
                previous_manifest.get(rel_path) == f_meta["file_hash"]
                and rel_path in chunks_by_file
            )
            if unchanged:
                all_chunks.extend(chunks_by_file[rel_path])
                reused_files += 1
                continue

            started = time.perf_counter()
            entities = MultiLanguageASTParser.parse_file(f_meta)
            parsed = time.perf_counter()
            chunks = HierarchicalCodeChunker.create_chunks(f_meta, entities)
            chunk_seconds += time.perf_counter() - parsed
            parse_seconds += parsed - started

            all_chunks.extend(chunks)
            changed_chunks.extend(chunks)
            parsed_files += 1

        # The reuse counts are the whole point of the incremental path, and they are the thing a
        # log line makes you go looking for and a trace just shows you.
        s.update(output={
            "files": file_count,
            "total_lines": total_lines,
            "languages": sorted(languages),
            "parsed_files": parsed_files,
            "reused_files": reused_files,
            "chunks": len(all_chunks),
            "changed_chunks": len(changed_chunks),
            "parse_seconds": round(parse_seconds, 3),
            "chunk_seconds": round(chunk_seconds, 3),
        })

    with tracing.span("prune_stale") as s:
        # Files that disappeared are simply absent from `discovered_files`, so their chunks fall
        # out of the index — but they must also be removed from the vector store, or deleted code
        # keeps being retrieved.
        surviving_ids = {chunk["chunk_id"] for chunk in all_chunks}
        stale_ids = [c["chunk_id"] for c in previous_chunks if c["chunk_id"] not in surviving_ids]
        if stale_ids:
            chroma_store.delete_chunks(repo_id, stale_ids)
        s.update(output={"stale_chunks_removed": len(stale_ids)})

    with tracing.span("embed") as s:
        # Only changed chunks need re-embedding — but only if the vector store actually still
        # holds the rest. The index on disk and the collection can diverge (the collection is
        # cleared, points elsewhere, or a write failed), and re-embedding just the changed files
        # would then leave semantic search querying a near-empty collection while the chunk index
        # looks complete. Verify the count and re-embed everything when it does not line up.
        embed_all = not previous_chunks
        resynced = False
        if not embed_all:
            try:
                present = chroma_store.get_or_create_collection(repo_id).count()
                if present < len(all_chunks):
                    logger.warning(
                        "Vector store for '%s' holds %d of %d chunks; re-embedding in full.",
                        repo_id, present, len(all_chunks),
                    )
                    embed_all = True
                    resynced = True
            except Exception as e:
                logger.warning("Could not verify vector store for '%s' (%s); re-embedding in full.", repo_id, e)
                embed_all = True
                resynced = True

        embedded = all_chunks if embed_all else changed_chunks
        chroma_store.add_chunks(repo_id, embedded)
        # `resynced` marks the fall-back to a full re-embed after the collection was found short.
        # It should be rare; if a dashboard shows it is not, the vector store is losing writes.
        s.update(output={"embedded_chunks": len(embedded), "embed_all": embed_all,
                         "resynced_after_drift": resynced})

    with tracing.span("graph") as s:
        neo4j_client.build_repository_graph(repo_id, all_chunks)
        s.update(output={"chunks": len(all_chunks)})

    with tracing.span("persist") as s:
        repository_chunks_cache[repo_id] = all_chunks
        index_store.save(repo_id, all_chunks)
        index_store.save_manifest(repo_id, manifest)
        BM25IndexCache.invalidate(repo_id)
        s.update(output={"chunks": len(all_chunks), "manifest_files": len(manifest)})

    if previous_chunks:
        logger.info(
            "Indexed '%s': %d file(s) reused, %d re-parsed, %d stale chunk(s) removed.",
            repo_id, reused_files, parsed_files, len(stale_ids),
        )

    return {
        "repository_id": repo_id,
        "name": repo_name,
        "file_count": file_count,
        "total_lines": total_lines,
        "languages": list(languages),
        # Reported rather than only logged: "a sync re-embeds only what changed" is the claim
        # the whole incremental path rests on, and a caller (or a test) needs to be able to
        # check it rather than take it on trust.
        "reused_files": reused_files,
        "parsed_files": parsed_files,
        "all_chunks": all_chunks
    }

@router.post("/upload", status_code=202)
async def upload_repository(file: UploadFile = File(...)):
    """
    Accepts a zip, extracts it, and indexes it in the background.

    Returns **202** with an id to poll rather than the finished result. The RETAIL upload took
    36 seconds, about 89 % of it embedding, and a browser or an intermediary proxy will abandon
    the request long before a large repository finishes.

    **Where the line falls matters.** The uploaded stream only exists for the duration of the
    request, so saving and extracting the archive has to happen before returning — that part is
    I/O-bound and bounded by `MAX_UPLOAD_BYTES`. Indexing is the slow half, and it is what becomes
    the job. Rejecting a corrupt or hostile archive therefore still produces a synchronous 400,
    which is the right answer for a request that was never going to work.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip repository archives are supported")

    repo_id = f"repo_{uuid.uuid4().hex[:8]}"
    repo_name = os.path.splitext(file.filename)[0]

    # Save permanently to data/repositories/{repo_id}
    repos_storage_dir = os.path.abspath(os.path.join(settings.DATA_DIR, "repositories"))
    os.makedirs(repos_storage_dir, exist_ok=True)

    target_repo_dir = os.path.join(repos_storage_dir, repo_id)
    os.makedirs(target_repo_dir, exist_ok=True)

    zip_path = os.path.join(target_repo_dir, file.filename)
    extract_dir = os.path.join(target_repo_dir, "codebase")

    # Extraction happens inline: it is fast, and its failures are the caller's fault, so they
    # belong in the response to the request that caused them.
    try:
        _stream_upload_to_disk(file, zip_path, settings.MAX_UPLOAD_BYTES)
        _safe_extract_zip(zip_path, extract_dir)
    except UnsafeArchiveError as e:
        purge_repository_stores(repo_id, target_repo_dir)
        shutil.rmtree(target_repo_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Rejected archive: {e}")
    except zipfile.BadZipFile:
        purge_repository_stores(repo_id, target_repo_dir)
        shutil.rmtree(target_repo_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive")
    except Exception as e:
        purge_repository_stores(repo_id, target_repo_dir)
        shutil.rmtree(target_repo_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process repository: {str(e)}")

    repo_col = db_client.get_collection("repositories")
    # The record is written before indexing starts so the repository is listable immediately and
    # `/status` has something to report against. The counts are filled in when the job finishes.
    await repo_col.insert_one({
        "repository_id": repo_id,
        "name": repo_name,
        "source": "zip",
        "file_count": 0,
        "total_lines": 0,
        "languages": [],
        "storage_path": extract_dir,
    })

    async def on_success(job_id: str, result: Dict[str, Any]) -> None:
        await repo_col.update_one(
            {"repository_id": job_id},
            {"$set": {
                "file_count": result["file_count"],
                "total_lines": result["total_lines"],
                "languages": result["languages"],
            }},
        )
        logger.info("Indexed upload '%s' (%d files).", repo_name, result["file_count"])

    async def on_failure(job_id: str, exc: BaseException) -> None:
        # Same reasoning as a failed GitHub import: a repository row that cannot answer anything
        # is worse than no row, and the reason stays readable through /status via the job runner.
        purge_repository_stores(job_id, target_repo_dir)
        shutil.rmtree(target_repo_dir, ignore_errors=True)
        await repo_col.delete_one({"repository_id": job_id})
        indexing_status[job_id] = f"failed: {exc}"
        logger.error("Indexing upload '%s' failed: %s", repo_name, exc)

    from app.core.jobs import indexing_jobs
    indexing_jobs.submit(
        repo_id,
        index_repository_folder, extract_dir, repo_id, repo_name,
        job_id=repo_id, trigger="upload",
        on_success=on_success,
        on_failure=on_failure,
    )
    indexing_status[repo_id] = "indexing"

    return {
        "repository_id": repo_id,
        "name": repo_name,
        "source": "zip",
        "status": "indexing",
    }

@router.get("", response_model=List[RepositoryResponse])
async def list_repositories():
    repo_col = db_client.get_collection("repositories")
    cursor = repo_col.find({})
    repos = await cursor.to_list(length=100)
    
    result = []
    for r in repos:
        result.append(RepositoryResponse(
            repository_id=r["repository_id"],
            name=r.get("name", r["repository_id"]),
            file_count=r.get("file_count", 0),
            total_lines=r.get("total_lines", 0),
            languages=r.get("languages", []),
            source=r.get("source", "zip"),
            branch=r.get("branch"),
            commit_sha=r.get("commit_sha"),
            synced_at=r.get("synced_at"),
        ))
    return result

@router.get("/{repository_id}/snippet")
async def get_snippet(repository_id: str, file_path: str, start_line: int = 1,
                      end_line: int = 0, at_sha: str = ""):
    """
    Returns the source for a cited range.

    Stored conversation turns hold citations as pointers — `(file_path, start_line, end_line)` —
    rather than inlined code, so a thread does not grow without bound. This is the other half of
    that arrangement: the UI resolves a pointer here when the reader expands a citation. Without
    it the code viewer worked during a live session and silently went inert after a reload,
    because the snippet had only ever existed in the in-flight response.

    `at_sha` is the commit the citing answer was written against. When the repository has since
    moved on, the pointer still resolves — but it resolves against *different code*: a citation
    recorded as `checkout.py:40-58` may now be a different function entirely. The result is
    flagged `stale` so the viewer can say so, rather than confidently showing the wrong thing,
    which is worse than showing nothing.

    Labelling rather than pinning is a deliberate trade. Pinning each thread to its own index
    would be exact and unaffordable: indexing every commit of a 500-commit repository is roughly
    428,500 chunks and ~16 GB, against ~5,300 shared chunks if the same code is stored once.
    """
    # Its own root trace: this is a request of its own, fired long after the answer that cited it,
    # so there is no query trace to hang it from. `match` is the reason it is worth tracing at all
    # — a citation that resolves as "nearest" is one the reader was shown slightly wrong code for,
    # and nothing else in the system would ever report that.
    async with tracing.start_trace(
        "snippet",
        metadata={"repository_id": repository_id, "file_path": file_path,
                  "start_line": start_line, "end_line": end_line, "cited_sha": at_sha or None},
        tags=["snippet"],
    ) as trace:
        with tracing.span("load_chunks", input={"repository_id": repository_id}) as s:
            chunks = get_repository_chunks(repository_id)
            s.update(output={"chunks": len(chunks), "indexed": bool(chunks)})

        if not chunks:
            raise HTTPException(
                status_code=409,
                detail=f"Repository '{repository_id}' is not indexed, so its source cannot be read.",
            )

        with tracing.span("resolve_pointer",
                          input={"file_path": file_path, "start_line": start_line,
                                 "end_line": end_line}) as s:
            candidates = [c for c in chunks if c.get("file_path") == file_path]
            if not candidates:
                s.update(output={"candidates": 0, "match": "no_such_file"})
                raise HTTPException(
                    status_code=404, detail=f"'{file_path}' is not part of this repository."
                )

            # Prefer the chunk that starts exactly where the citation says; otherwise the smallest
            # chunk that encloses the range, which is the tightest definition containing those
            # lines.
            resolved = next((c for c in candidates if c["start_line"] == start_line), None)
            match = "exact" if resolved is not None else ""
            if resolved is None and end_line:
                enclosing = [c for c in candidates
                             if c["start_line"] <= start_line and c["end_line"] >= end_line]
                if enclosing:
                    resolved = min(enclosing, key=lambda c: c["end_line"] - c["start_line"])
                    match = "enclosing"
            if resolved is None:
                resolved = min(candidates, key=lambda c: abs(c["start_line"] - start_line))
                match = "nearest"
            chunk = resolved

            s.update(output={"candidates": len(candidates), "match": match,
                             "symbol": chunk.get("symbol", ""),
                             "resolved_start": chunk["start_line"],
                             "resolved_end": chunk["end_line"],
                             "line_offset": chunk["start_line"] - start_line})

        payload = {
            "repository_id": repository_id,
            "file_path": chunk["file_path"],
            "symbol": chunk.get("symbol", ""),
            "language": chunk.get("language", ""),
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "code": chunk.get("code_snippet", ""),
            "stale": False,
        }

        if at_sha:
            with tracing.span("staleness_check", input={"cited_sha": at_sha}) as s:
                repo = await db_client.get_collection("repositories").find_one(
                    {"repository_id": repository_id}
                )
                current_sha = (repo or {}).get("commit_sha") or ""
                if current_sha and current_sha != at_sha:
                    payload["stale"] = True
                    payload["indexed_sha"] = current_sha
                    payload["cited_sha"] = at_sha
                    # Whether the citation still lands where it did is knowable cheaply: if the
                    # pointer resolved to a chunk starting on the cited line, the symbol has not
                    # moved, so the reader can be told the difference between "shifted" and
                    # "probably fine".
                    payload["moved"] = chunk["start_line"] != start_line
                s.update(output={"indexed_sha": current_sha, "stale": payload["stale"],
                                 "moved": payload.get("moved"),
                                 # No recorded sha means the repository came from a zip, where
                                 # staleness is not knowable — distinct from "checked and fresh".
                                 "comparable": bool(current_sha)})

        trace.update(output={"match": match, "stale": payload["stale"],
                             "symbol": payload["symbol"],
                             "lines": f"{payload['start_line']}-{payload['end_line']}"})
        return payload


@router.get("/{repository_id}/status")
async def get_repository_status(repository_id: str):
    """
    Reports whether a repository's index is ready, so clients can wait rather than guess.

    Also consults the background job runner. A GitHub import that fails deletes its repository
    record — a row that cannot answer anything is worse than no row — so the job is the only
    remaining account of what went wrong, and a client polling this endpoint deserves the reason
    rather than a bare 404.
    """
    from app.core.jobs import indexing_jobs

    repo = await db_client.get_collection("repositories").find_one({"repository_id": repository_id})
    job = indexing_jobs.get(repository_id)

    if not repo:
        if job is not None:
            return {
                "repository_id": repository_id,
                "name": repository_id,
                "status": job.state if job.state != "done" else "ready",
                "chunk_count": 0,
                "index_persisted": False,
                "error": job.error,
                "elapsed_seconds": round(job.elapsed, 2),
            }
        raise HTTPException(status_code=404, detail="Repository not found")

    if job is not None and not job.is_terminal:
        state = "indexing"
    else:
        state = indexing_status.get(repository_id)
        if state is None:
            state = "ready" if index_store.exists(repository_id) else "not indexed"

    payload = {
        "repository_id": repository_id,
        "name": repo.get("name", repository_id),
        "status": state,
        "chunk_count": len(repository_chunks_cache.get(repository_id, [])),
        "index_persisted": index_store.exists(repository_id),
    }
    if job is not None:
        payload["elapsed_seconds"] = round(job.elapsed, 2)
        if job.error:
            payload["error"] = job.error
    # Git repositories carry their upstream identity; the client needs it to choose /sync over
    # /reindex, and to show which commit an answer was based on.
    for field in ("source", "branch", "commit_sha", "synced_at", "clone_url"):
        if repo.get(field):
            payload[field] = repo[field]
    return payload


@router.post("/{repository_id}/reindex")
async def reindex_repository(repository_id: str, full: bool = False):
    """
    Re-indexes a repository from its stored source.

    Incremental by default: files whose content hash is unchanged keep their existing chunks and
    only changed or new files are re-parsed (F-23). Pass `full=true` to force a complete rebuild,
    which is what you want if the index itself is suspect rather than the source.
    """
    repo = await db_client.get_collection("repositories").find_one({"repository_id": repository_id})
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    source_path = repo.get("storage_path")
    if not source_path or not os.path.exists(source_path):
        # For a GitHub import there is deliberately no stored source — the clone is deleted once
        # it has been indexed — so "upload it again" is the wrong advice and re-acquiring the
        # source is a single call away. Point at it rather than sending the user round a loop.
        if repo.get("source") == "github":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{repo.get('name', repository_id)}' was imported from GitHub and keeps no "
                    f"local source. Use POST /repositories/{repository_id}/sync, which re-clones "
                    f"before re-indexing."
                ),
            )
        raise HTTPException(
            status_code=409,
            detail=(
                f"Source for '{repo.get('name', repository_id)}' is no longer on disk "
                f"({source_path or 'no path recorded'}). Upload the repository again."
            ),
        )

    if indexing_status.get(repository_id) == "indexing":
        raise HTTPException(status_code=409, detail="This repository is already being indexed.")

    try:
        result = await asyncio.to_thread(
            index_repository_folder, source_path, repository_id,
            repo.get("name", repository_id), full, job_id=repository_id, trigger="reindex",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-index failed: {e}")

    await db_client.get_collection("repositories").update_one(
        {"repository_id": repository_id},
        {"$set": {
            "file_count": result["file_count"],
            "total_lines": result["total_lines"],
            "languages": result["languages"],
        }},
    )

    return {
        "repository_id": repository_id,
        "status": "ready",
        "mode": "full" if full else "incremental",
        "file_count": result["file_count"],
        "chunk_count": len(result["all_chunks"]),
    }


@router.delete("/{repository_id}")
async def delete_repository(repository_id: str, delete_conversations: bool = False):
    """
    Deletes a repository and everything derived from it.

    Conversations are **kept by default**. A thread is the only thing here that cannot be
    regenerated — the index, embeddings and graph all rebuild from source in seconds, while a
    conversation exists nowhere else. So the default is to report how many threads are affected
    and let the caller decide, rather than to quietly destroy them alongside derived data.

    Pass `delete_conversations=true` to remove them too.
    """
    repo_col = db_client.get_collection("repositories")
    repo = await repo_col.find_one({"repository_id": repository_id})
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    if indexing_status.get(repository_id) == "indexing":
        raise HTTPException(
            status_code=409,
            detail="This repository is currently being indexed. Wait for it to finish, then delete.",
        )

    from app.memory.conversation_memory import ConversationMemory
    threads = await ConversationMemory.list_conversations(repository_id=repository_id)

    result = purge_repository_stores(repository_id, repo.get("storage_path"))
    await repo_col.delete_one({"repository_id": repository_id})
    result["record_deleted"] = True

    if delete_conversations:
        deleted = 0
        for thread in threads:
            if await ConversationMemory.delete_conversation(thread["conversation_id"]):
                deleted += 1
        result["conversations_deleted"] = deleted
        result["conversations_orphaned"] = 0
    else:
        result["conversations_deleted"] = 0
        # Named plainly: these threads survive but can no longer answer or resolve a citation,
        # because the index they point at is gone.
        result["conversations_orphaned"] = len(threads)

    logger.info("Deleted repository '%s': %s", repository_id, result)
    return result


@router.get("/{repository_id}", response_model=RepositoryResponse)
async def get_repository(repository_id: str):
    repo_col = db_client.get_collection("repositories")
    repo = await repo_col.find_one({"repository_id": repository_id})
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return RepositoryResponse(
        repository_id=repo["repository_id"],
        name=repo.get("name", repo["repository_id"]),
        file_count=repo.get("file_count", 0),
        total_lines=repo.get("total_lines", 0),
        languages=repo.get("languages", []),
        source=repo.get("source", "zip"),
        branch=repo.get("branch"),
        commit_sha=repo.get("commit_sha"),
        synced_at=repo.get("synced_at"),
    )
