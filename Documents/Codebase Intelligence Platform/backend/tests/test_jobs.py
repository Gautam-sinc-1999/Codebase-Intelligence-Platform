"""
Background job execution.

The load-bearing test here is `test_drain_waits_for_threads_rather_than_cancelling`. Everything
else is ordinary bookkeeping; that one encodes why the module exists at all.
"""
import asyncio
import threading
import time

import pytest

async def _await_event(event: threading.Event, timeout: float = 5.0) -> bool:
    """
    Waits for a worker thread's event without blocking the event loop.

    `threading.Event.wait()` called directly from a test would stall the loop that the job
    runner's supervising coroutine needs in order to make progress — the test would then be
    measuring its own deadlock.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if event.is_set():
            return True
        await asyncio.sleep(0.02)
    return event.is_set()


from app.core.jobs import (
    JobRunner,
    JobAlreadyRunning,
    QUEUED,
    RUNNING,
    DONE,
    FAILED,
)


async def test_submit_returns_before_the_work_finishes():
    """The premise of the whole module: the caller is not held for the duration."""
    release = threading.Event()
    runner = JobRunner(max_workers=1, name="t")

    def slow():
        release.wait(timeout=5)
        return "finished"

    started = time.time()
    job = runner.submit("j1", slow)
    assert time.time() - started < 0.5
    assert not job.is_terminal

    release.set()
    await runner.drain()
    assert runner.get("j1").state == DONE


async def test_job_states_progress_in_order():
    entered = threading.Event()
    release = threading.Event()
    runner = JobRunner(max_workers=1, name="t")

    def work():
        entered.set()
        release.wait(timeout=5)
        return 42

    job = runner.submit("j1", work)
    assert await _await_event(entered)
    assert job.state == RUNNING and job.started_at is not None

    release.set()
    await runner.drain()
    assert job.state == DONE and job.result == 42 and job.finished_at is not None
    assert job.elapsed > 0


async def test_drain_waits_for_threads_rather_than_cancelling():
    """
    F-39, encoded.

    `run_in_executor` hands work to an OS thread that cannot be interrupted. Cancelling the
    awaitable returns at once while that thread keeps writing — to ChromaDB, to Neo4j, to the
    index files. A shutdown that cancels therefore reports success while corrupting state.

    The assertion is not that `drain()` returned; it is that the *thread's own final statement
    ran* before it did.
    """
    finished_writing = threading.Event()
    runner = JobRunner(max_workers=1, name="t")

    def writes_to_disk():
        time.sleep(0.4)
        # Stands in for the last write of an index: if drain cancels, this never happens.
        finished_writing.set()
        return "written"

    runner.submit("writer", writes_to_disk)
    completed = await runner.drain(timeout=10)

    assert completed is True
    assert finished_writing.is_set(), "drain returned while the worker thread was still running"
    assert runner.get("writer").state == DONE


async def test_cancelling_would_not_have_waited():
    """
    The counter-example, so the reason for `drain`'s design is demonstrated rather than asserted.

    Cancelling the supervising task returns immediately and the thread carries on — which is
    precisely the failure mode `drain` avoids.
    """
    entered = threading.Event()
    still_running = threading.Event()
    runner = JobRunner(max_workers=1, name="t")

    def slow():
        entered.set()
        time.sleep(0.5)
        still_running.set()
        return "done anyway"

    runner.submit("victim", slow)
    task = next(iter(runner._tasks.values()))

    # Cancel only once the thread is demonstrably executing. Cancelling earlier would cancel the
    # executor future before a worker picked it up — the work would never start, which proves the
    # opposite of the point being made here.
    assert await _await_event(entered), "precondition: the worker thread must be running"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The thread is demonstrably unaffected by the cancellation.
    assert not still_running.is_set(), "precondition: the thread should not have finished yet"
    assert await _await_event(still_running), \
        "the thread kept running after cancellation, as expected"

    await runner.drain()


async def test_failure_is_recorded_and_reported_to_the_callback():
    seen = {}

    async def on_failure(job_id, exc):
        seen["job_id"] = job_id
        seen["error"] = str(exc)

    runner = JobRunner(max_workers=1, name="t")

    def explodes():
        raise ValueError("index write failed")

    runner.submit("bad", explodes, on_failure=on_failure)
    await runner.drain()

    job = runner.get("bad")
    assert job.state == FAILED
    assert "index write failed" in job.error
    assert seen == {"job_id": "bad", "error": "index write failed"}


async def test_success_callback_receives_the_result():
    seen = {}

    async def on_success(job_id, result):
        seen[job_id] = result

    runner = JobRunner(max_workers=1, name="t")
    runner.submit("good", lambda: {"chunks": 18}, on_success=on_success)
    await runner.drain()

    assert seen == {"good": {"chunks": 18}}


async def test_plain_function_callbacks_are_supported_too():
    seen = []
    runner = JobRunner(max_workers=1, name="t")
    runner.submit("j", lambda: 1, on_success=lambda jid, res: seen.append((jid, res)))
    await runner.drain()
    assert seen == [("j", 1)]


async def test_a_second_submission_for_a_live_job_is_refused():
    """Two indexes of one repository would race each other through the same stores."""
    release = threading.Event()
    runner = JobRunner(max_workers=1, name="t")

    runner.submit("repo1", lambda: release.wait(timeout=5))
    with pytest.raises(JobAlreadyRunning):
        runner.submit("repo1", lambda: None)

    release.set()
    await runner.drain()


async def test_a_finished_job_id_can_be_reused():
    runner = JobRunner(max_workers=1, name="t")
    runner.submit("repo1", lambda: "first")
    await runner.drain()

    runner.submit("repo1", lambda: "second")
    await runner.drain()
    assert runner.get("repo1").result == "second"


async def test_concurrency_is_capped():
    """
    Embedding is CPU-bound and already internally parallel, so more concurrent indexes make
    every one of them slower rather than any of them faster.
    """
    concurrent = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()
    runner = JobRunner(max_workers=2, name="t")

    def work():
        nonlocal concurrent, peak
        with lock:
            concurrent += 1
            peak = max(peak, concurrent)
        release.wait(timeout=5)
        with lock:
            concurrent -= 1

    for i in range(4):
        runner.submit(f"j{i}", work)

    # Give the executor a moment to saturate its two workers.
    await asyncio.sleep(0.3)
    assert runner.get("j3").state == QUEUED, "a job ran before a worker was free"

    release.set()
    await runner.drain()
    assert peak <= 2, f"{peak} jobs ran at once with max_workers=2"


async def test_active_jobs_lists_only_unfinished_work():
    release = threading.Event()
    runner = JobRunner(max_workers=2, name="t")

    runner.submit("done_soon", lambda: "x")
    await asyncio.sleep(0.2)
    runner.submit("still_going", lambda: release.wait(timeout=5))
    await asyncio.sleep(0.1)

    active = {j.job_id for j in runner.active_jobs()}
    assert active == {"still_going"}
    assert runner.is_active("still_going") and not runner.is_active("done_soon")

    release.set()
    await runner.drain()


async def test_forget_drops_finished_jobs_but_not_live_ones():
    release = threading.Event()
    runner = JobRunner(max_workers=2, name="t")

    runner.submit("finished", lambda: "x")
    runner.submit("live", lambda: release.wait(timeout=5))
    await asyncio.sleep(0.2)

    runner.forget("finished")
    runner.forget("live")

    assert runner.get("finished") is None
    assert runner.get("live") is not None, "forgot a job that was still running"

    release.set()
    await runner.drain()


async def test_draining_an_idle_runner_is_a_noop():
    runner = JobRunner(max_workers=1, name="t")
    assert await runner.drain() is True


async def test_submissions_are_refused_while_draining():
    """
    The guard that matters is refusing work *during* the wait — accepting a job then is accepting
    work nothing will wait for. Once the drain is over there is nothing in flight, so the runner
    is idle rather than dead.
    """
    release = threading.Event()
    runner = JobRunner(max_workers=1, name="t")
    runner.submit("live", lambda: release.wait(timeout=5))

    draining = asyncio.ensure_future(runner.drain(timeout=10))
    await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError):
        runner.submit("during_drain", lambda: None)

    release.set()
    await draining


async def test_a_drained_runner_can_be_used_again():
    """A module-level runner outlives one application lifespan — the suite starts the app often."""
    runner = JobRunner(max_workers=1, name="t")
    runner.submit("first", lambda: "a")
    await runner.drain()

    runner.submit("second", lambda: "b")
    await runner.drain()
    assert runner.get("second").result == "b"


async def test_drain_reports_incompletion_but_still_waits_for_the_thread():
    """
    A missed deadline bounds how long the caller blocks — it never downgrades the guarantee.
    The executor is still shut down with wait=True, so the thread finishes regardless.
    """
    wrote = threading.Event()
    runner = JobRunner(max_workers=1, name="t")

    def slow():
        time.sleep(0.6)
        wrote.set()

    runner.submit("slow", slow)
    completed = await runner.drain(timeout=0.1)

    assert completed is False, "expected drain to report the deadline was missed"
    assert wrote.is_set(), "drain abandoned a running thread instead of waiting for it"


async def test_app_shutdown_drains_indexing_jobs():
    """
    The guarantee, end to end through the real application lifespan.

    Driven through `lifespan` itself rather than a TestClient, so the assertion is about the
    production shutdown path. It matters here specifically because ChromaDB is closed moments
    later, and tearing its client out from under a live upsert fails with
    'RustBindingsAPI object has no attribute bindings'.
    """
    import app.main as main_module
    from app.core.jobs import indexing_jobs
    from app.core.config import settings

    wrote = threading.Event()

    # Long enough that the remainder of shutdown (closing the HTTP client and ChromaDB) cannot
    # coincidentally outlast it — otherwise the test would pass with no drain at all.
    def slow_write():
        time.sleep(2.0)
        wrote.set()
        return "indexed"

    # Bootstrap indexes the sample repository under the fixed id 'sample_ecommerce_repo' —
    # the same id a developer's own copy uses. Neo4j Community has one database, so letting
    # bootstrap run here would write to the real graph. The api_client fixture disables it for
    # exactly this reason; driving the lifespan directly means doing so explicitly.
    original_sample_dir = settings.SAMPLE_REPO_DIR
    settings.SAMPLE_REPO_DIR = "/nonexistent-so-bootstrap-is-a-noop"
    try:
        async with main_module.lifespan(main_module.app):
            indexing_jobs.submit("shutdown_probe", slow_write)
            assert not wrote.is_set(), "precondition: the write should still be in flight"

        assert wrote.is_set(), \
            "app shutdown returned while an indexing thread was still writing"
        assert indexing_jobs.get("shutdown_probe").state == DONE
    finally:
        settings.SAMPLE_REPO_DIR = original_sample_dir
        indexing_jobs._jobs.pop("shutdown_probe", None)
