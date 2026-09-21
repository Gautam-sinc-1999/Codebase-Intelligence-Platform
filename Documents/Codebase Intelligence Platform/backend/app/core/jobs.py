"""
Background job execution for work too slow to hold an HTTP request open.

Indexing a repository takes tens of seconds — 36 s for the RETAIL upload, ~89 % of it spent
embedding — and both a browser and an intermediary proxy will give up long before a large
repository finishes. The request therefore has to return a handle immediately and let the client
poll, which is what this module provides.

**Cancellation is not how this shuts down, and that is the whole point.**

`asyncio.to_thread` (and `run_in_executor`) hand work to an OS thread. Cancelling the awaitable
returns control to the event loop at once, but the thread keeps running to completion — it is not
interruptible. A shutdown path that cancels therefore *reports* a clean stop while a live thread
is still writing embeddings to ChromaDB, nodes to Neo4j and JSON to disk, producing exactly the
half-written state that shutdown existed to prevent. This cost us F-39 once already.

So `drain()` waits. A dedicated executor is used rather than the default one precisely because
`ThreadPoolExecutor.shutdown(wait=True)` expresses that guarantee directly, and because indexing
must not queue behind unrelated `to_thread` calls elsewhere in the application.
"""
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("core.jobs")

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

TERMINAL_STATES = (DONE, FAILED)


@dataclass
class Job:
    """One unit of background work, addressed by a caller-chosen id."""
    job_id: str
    state: str = QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def as_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "elapsed_seconds": round(self.elapsed, 2),
            "error": self.error,
        }


class JobAlreadyRunning(Exception):
    """A job with this id is queued or running; starting a second would race the first."""


class JobRunner:
    """
    Runs CPU-bound callables off the event loop, one record per job id.

    `max_workers` is deliberately small. The expensive phase is embedding, which is CPU-bound and
    already internally parallel, so admitting many concurrent indexes does not make any of them
    faster — it makes all of them slower and multiplies peak memory.
    """

    def __init__(self, max_workers: int = 2, name: str = "jobs"):
        self.max_workers = max_workers
        self._name = name
        self._jobs: Dict[str, Job] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix=self._name
            )
            self._closed = False
        return self._executor

    def submit(
        self,
        job_id: str,
        fn: Callable[..., Any],
        *args,
        on_success: Optional[Callable[[str, Any], Any]] = None,
        on_failure: Optional[Callable[[str, BaseException], Any]] = None,
        **kwargs,
    ) -> Job:
        """
        Schedules `fn(*args, **kwargs)` on a worker thread and returns immediately.

        `on_success` / `on_failure` run on the event loop once the thread finishes, so they may be
        coroutines — which is what lets the caller record a result in Mongo, or purge the stores a
        failed index left behind, without either happening on a worker thread.
        """
        if self._closed:
            raise RuntimeError("Job runner has been shut down.")

        existing = self._jobs.get(job_id)
        if existing and not existing.is_terminal:
            raise JobAlreadyRunning(f"Job '{job_id}' is already {existing.state}.")

        job = Job(job_id=job_id)
        self._jobs[job_id] = job

        loop = asyncio.get_running_loop()
        executor = self._ensure_executor()

        def run() -> Any:
            # Runs on the worker thread. State flips to RUNNING here rather than at submit time,
            # so a job waiting for a free worker is honestly reported as queued.
            job.state = RUNNING
            job.started_at = time.time()
            return fn(*args, **kwargs)

        # Handed to the executor here, not inside `supervise`, so the worker thread starts the
        # moment submit() is called rather than whenever the event loop next yields. A caller that
        # blocks before its next await would otherwise have scheduled nothing at all.
        future = loop.run_in_executor(executor, run)

        async def supervise() -> Any:
            try:
                result = await future
            except BaseException as exc:
                job.state = FAILED
                job.finished_at = time.time()
                job.error = str(exc)
                logger.error("Job '%s' failed after %.1fs: %s", job_id, job.elapsed, exc)
                if on_failure is not None:
                    await _maybe_await(on_failure(job_id, exc))
                return None

            job.state = DONE
            job.finished_at = time.time()
            job.result = result
            logger.info("Job '%s' finished in %.1fs.", job_id, job.elapsed)
            if on_success is not None:
                await _maybe_await(on_success(job_id, result))
            return result

        task = asyncio.ensure_future(supervise())
        self._tasks[job_id] = task
        task.add_done_callback(lambda _t, jid=job_id: self._tasks.pop(jid, None))
        return job

    # ------------------------------------------------------------------ inspection

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def is_active(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        return job is not None and not job.is_terminal

    def active_jobs(self) -> List[Job]:
        return [j for j in self._jobs.values() if not j.is_terminal]

    def forget(self, job_id: str) -> None:
        """Drops a finished job's record — called when its repository is deleted."""
        job = self._jobs.get(job_id)
        if job is not None and job.is_terminal:
            self._jobs.pop(job_id, None)

    # ------------------------------------------------------------------ shutdown

    async def drain(self, timeout: Optional[float] = 60.0) -> bool:
        """
        Waits for in-flight work to finish. Returns True if everything completed.

        **Deliberately does not cancel.** Cancelling would return immediately while the worker
        threads carried on writing to ChromaDB, Neo4j and the index files — the exact race this
        design exists to avoid (F-39). Waiting is the only way to be sure that what is on disk
        when the process exits is consistent.

        On timeout the executor is still shut down with `wait=True`, so the guarantee holds even
        when the deadline is missed; the timeout only bounds how long the *caller* blocks before
        being told the wait is running long.
        """
        self._closed = True
        pending = [t for t in self._tasks.values() if not t.done()]

        completed = True
        if pending:
            logger.info("Waiting for %d in-flight job(s) to finish.", len(pending))
            done, still_running = await asyncio.wait(pending, timeout=timeout)
            if still_running:
                completed = False
                logger.warning(
                    "%d job(s) still running after %.0fs; waiting for their threads anyway "
                    "rather than leaving partial writes behind.",
                    len(still_running), timeout or 0,
                )

        if self._executor is not None:
            # wait=True is the guarantee: threads run to completion before the process exits.
            self._executor.shutdown(wait=True)
            self._executor = None

        # Drained means idle, not dead. `_closed` blocks submissions *during* the wait, so work
        # can never be accepted that nothing will wait for; once the wait is over there is
        # nothing in flight and the runner is usable again. A runner that stayed closed would
        # survive exactly one application lifespan, which is wrong for a module-level singleton
        # — the test suite starts the app many times in one process, and so can a reload.
        self._closed = False
        return completed


async def _maybe_await(value: Any) -> Any:
    """Callbacks may be plain functions or coroutines; both are supported."""
    if asyncio.iscoroutine(value):
        return await value
    return value


# The application-wide runner for repository indexing.
indexing_jobs = JobRunner(max_workers=2, name="indexer")
