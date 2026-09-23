import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.security import require_api_key, is_authenticated
from app.core.database import db_client, redis_client
from app.api.repositories import (
    router as repos_router,
    repository_chunks_cache,
    index_repository_folder,
    get_repository_chunks,
    reconcile_repositories,
)
from app.ingestion.index_store import index_store
from app.memory.thread_migration import migrate_threads_to_mongo
from app.api.github import router as github_router
from app.api.conversations import router as convs_router
from app.api.graph import router as graph_router

from app.ingestion.discovery import FileDiscovery
from app.ingestion.ast_parser import MultiLanguageASTParser
from app.ingestion.chunker import HierarchicalCodeChunker
from app.vector.chromadb_client import chroma_store
from app.core.jobs import indexing_jobs
from app.observability import shutdown_tracing, tracing_enabled
from app.graph.neo4j_client import neo4j_client
from app.agents.orchestrator import close_http_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown, as one context manager.

    Replaces the deprecated @app.on_event hooks. Everything before `yield` runs at startup and
    everything after runs at shutdown, which keeps paired setup and teardown — the HTTP client
    in particular — visible together instead of in two unrelated handlers.
    """
    logger.info("Initializing Codebase Intelligence Platform Services...")

    if settings.API_KEY:
        logger.info("API key authentication is ENABLED (header: %s).", settings.API_KEY_HEADER)
    else:
        logger.warning(
            "API key authentication is DISABLED - every endpoint is open to anyone who can reach "
            "this port. Set API_KEY before exposing this service beyond localhost."
        )
    if settings.cors_allows_any_origin:
        logger.warning("CORS is configured to allow any origin; credentials are disabled as a result.")

    await db_client.connect()
    await redis_client.connect()

    # Indexing runs in the background. It is CPU-bound and previously ran inline here, blocking
    # the event loop for as long as it took to re-parse every repository on disk — on every boot,
    # because nothing was persisted. The server now accepts requests immediately; a repository
    # whose index is not ready yet answers 409 (see conversations.send_message) rather than
    # pretending to have no matching code.
    bootstrap_task = asyncio.create_task(_bootstrap_indexes())

    yield

    # Wait for background indexing rather than cancelling it.
    #
    # Indexing runs via asyncio.to_thread, and cancelling a task that is awaiting a thread
    # returns as soon as the future is cancelled — the worker thread carries on regardless.
    # Closing the vector store at that point tears the client out from under a live upsert,
    # which fails with 'RustBindingsAPI object has no attribute bindings'. Since the thread
    # cannot be interrupted anyway, cancelling achieves nothing; waiting is what makes the
    # subsequent close safe.
    if not bootstrap_task.done():
        logger.info("Waiting for background indexing to finish before shutdown...")
    try:
        await asyncio.wait_for(
            asyncio.shield(bootstrap_task), timeout=settings.SHUTDOWN_GRACE_SECONDS
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Index bootstrap did not finish within %ss; closing stores anyway. An in-flight "
            "index may report an error as a result.",
            settings.SHUTDOWN_GRACE_SECONDS,
        )
    except Exception as e:
        logger.error("Index bootstrap ended with an error: %s", e)

    # Background indexing jobs write to ChromaDB, Neo4j and the index files, so they must be
    # drained before those stores are closed — for exactly the reason described above, and by
    # the same means: waiting, never cancelling.
    if indexing_jobs.active_jobs():
        logger.info(
            "Waiting for %d background indexing job(s) to finish...",
            len(indexing_jobs.active_jobs()),
        )
    if not await indexing_jobs.drain(timeout=settings.SHUTDOWN_GRACE_SECONDS):
        logger.warning(
            "Indexing jobs exceeded the %ss grace period; their threads were still waited on "
            "so nothing is left half-written.",
            settings.SHUTDOWN_GRACE_SECONDS,
        )

    # Flushed before the stores close: a queued trace dropped at exit is a trace of exactly the
    # shutdown you wanted to look at.
    shutdown_tracing()

    await close_http_client()
    # Closed deterministically rather than at interpreter teardown, where it can abort the
    # process (F-36).
    chroma_store.close()
    logger.info("Shut down cleanly.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# allow_origins=["*"] with allow_credentials=True is rejected outright by browsers, so the
# previous configuration was both maximally permissive in intent and non-functional in practice.
# Credentials are enabled only alongside an explicit origin list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=not settings.cors_allows_any_origin,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", settings.API_KEY_HEADER],
)

# Every API route is guarded. With API_KEY unset the dependency is a no-op, so local
# development is unchanged; setting it turns auth on across all three routers at once.
api_guard = [Depends(require_api_key)]

# Registered before the repositories router: its paths are literal ('/from-github',
# '/github/branches') and would otherwise be captured by that router's '/{repository_id}'.
app.include_router(github_router, prefix=settings.API_PREFIX, dependencies=api_guard)
app.include_router(repos_router, prefix=settings.API_PREFIX, dependencies=api_guard)
app.include_router(convs_router, prefix=settings.API_PREFIX, dependencies=api_guard)
app.include_router(graph_router, prefix=settings.API_PREFIX, dependencies=api_guard)



async def _bootstrap_indexes():
    """Registers the sample repository and ensures every stored repository has a usable index."""
    try:
        sample_dir = settings.SAMPLE_REPO_DIR
        if os.path.exists(sample_dir):
            repo_id = "sample_ecommerce_repo"
            repo_col = db_client.get_collection("repositories")

            if index_store.exists(repo_id) and get_repository_chunks(repo_id):
                logger.info("Sample repository loaded from its persisted index.")
            else:
                logger.info("Indexing sample repository at %s...", sample_dir)
                index_res = await asyncio.to_thread(
                    index_repository_folder, sample_dir, repo_id, "sample-ecommerce-product"
                )
                if not await repo_col.find_one({"repository_id": repo_id}):
                    await repo_col.insert_one({
                        "repository_id": repo_id,
                        "name": "sample-ecommerce-product",
                        "file_count": index_res["file_count"],
                        "total_lines": index_res["total_lines"],
                        "languages": index_res["languages"],
                        "storage_path": sample_dir
                    })
                logger.info(
                    "Sample repository indexed (%d files, %d chunks).",
                    index_res["file_count"], len(index_res["all_chunks"]),
                )

        # Uploaded repositories: load the persisted index where one exists, and only re-parse
        # from source when it does not.
        repos_storage_dir = os.path.abspath(os.path.join(settings.DATA_DIR, "repositories"))
        if os.path.exists(repos_storage_dir):
            for r_dir in sorted(os.listdir(repos_storage_dir)):
                r_path = os.path.join(repos_storage_dir, r_dir, "codebase")
                if not os.path.exists(r_path) or r_dir in repository_chunks_cache:
                    continue

                if get_repository_chunks(r_dir):
                    continue

                logger.info("Indexing uploaded repository '%s' from source...", r_dir)
                try:
                    await asyncio.to_thread(index_repository_folder, r_path, r_dir, f"Repo-{r_dir[:8]}")
                except Exception as e:
                    logger.error("Could not index '%s': %s", r_dir, e)

        # Conversations moved from folder-per-thread into MongoDB. Idempotent, and it never
        # deletes the folders — threads are the one thing here that cannot be rebuilt.
        migrated = await migrate_threads_to_mongo()
        if migrated["migrated"]:
            logger.info("Migrated %d conversation(s) to MongoDB.", len(migrated["migrated"]))
        if migrated["failed"]:
            logger.warning("Could not migrate: %s", ", ".join(migrated["failed"]))

        # Removes records that can no longer become usable. Run last, so anything the steps
        # above were able to restore is already in place and is not mistaken for stale.
        reconciled = await reconcile_repositories()
        if reconciled["removed"]:
            logger.warning(
                "Removed %d unusable repository record(s): %s",
                len(reconciled["removed"]), ", ".join(reconciled["removed"]),
            )

        logger.info("Index bootstrap complete.")
    except Exception as e:
        logger.error("Index bootstrap failed: %s", e)

@app.get("/")
async def root(request: Request):
    """
    Liveness endpoint. Stays unauthenticated so it is usable as a health check, but the backend
    breakdown is reconnaissance once the service is exposed, so it is shown only to an
    authenticated caller (or to everyone when auth is disabled, where there is no distinction).
    """
    payload = {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "auth": "enabled" if settings.API_KEY else "disabled",
    }

    if is_authenticated(request):
        # Reporting which backend each store actually resolved to makes degraded operation
        # visible. Every store falls back silently when its server is unreachable, so without
        # this the system looks identical whether it is durably persisting data or holding it
        # in a process dict.
        backends = {
            "documents": db_client.backend_name,
            "cache": redis_client.backend_name,
            "vectors": "chromadb-fallback" if chroma_store.is_fallback else "chromadb",
            "graph": "networkx-fallback" if neo4j_client.is_fallback else "neo4j",
        }
        payload["backends"] = backends
        payload["tracing"] = "langfuse" if tracing_enabled() else "off"
        payload["degraded"] = [name for name, backend in backends.items() if "fallback" in backend]

        # Why, not just what. A missing client library and an unreachable server look identical
        # from here, and they have completely different fixes — one is a pip install, the other
        # is a server to start. Reporting only "degraded" sends people to the wrong box.
        reasons = {}
        if getattr(db_client, "degraded_reason", None):
            reasons["documents"] = db_client.degraded_reason
        if getattr(redis_client, "unavailable_reason", None):
            reasons["cache"] = redis_client.unavailable_reason
        if reasons:
            payload["degraded_reasons"] = reasons

    return payload
