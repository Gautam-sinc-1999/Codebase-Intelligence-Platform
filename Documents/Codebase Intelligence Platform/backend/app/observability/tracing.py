"""
Langfuse tracing, wrapped so that observability can never break the product.

Every function here is a no-op when Langfuse is not installed or not configured, and every call
into the SDK is guarded. That is not defensive habit — it is the same rule the rest of this system
follows for MongoDB, Redis, Neo4j and ChromaDB: **a store being unavailable degrades the feature
that store serves, and nothing else.** Tracing serves the developer looking at a dashboard. An
unreachable dashboard must not cost a user their answer.

Usage mirrors the SDK but tolerates its absence:

    async with start_trace("query", session_id=conversation_id) as t:
        with span("retrieve") as s:
            ...
            s.update(output={"chunks": 6})
        with generation("llm", model=model, input=messages) as g:
            ...
            g.update(output=text, usage_details={"input": 1900, "output": 200})
        score("citation_validity", 0.83)

Configuration comes from the environment, so an unconfigured deployment simply does not trace:

    LANGFUSE_PUBLIC_KEY   LANGFUSE_SECRET_KEY   LANGFUSE_HOST
"""
import logging
import os
import sys
from contextlib import ExitStack, contextmanager, asynccontextmanager
from typing import Any, Dict, Optional

# Imported for its side effect: `app.core.config` is what reads `backend/.env` into the
# environment, and the keys below are read with `os.getenv`. Without this, whether tracing works
# depends on whether something *else* happened to import config first — the application does, so
# the server traced correctly while a standalone script importing only this module silently found
# no keys and reported "tracing is off".
from app.core import config as _config  # noqa: F401

logger = logging.getLogger("observability.tracing")

_client = None
_init_attempted = False


def _resolve_client():
    """
    Returns a Langfuse client, or None.

    Initialised once and cached, including the failure. A misconfigured deployment should log one
    line at startup, not one line per query.
    """
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True

    public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public or not secret:
        logger.info("Langfuse keys not set; tracing is off.")
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=public,
            secret_key=secret,
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
        )
        logger.info("Langfuse tracing enabled (host: %s).",
                    os.getenv("LANGFUSE_HOST", "http://localhost:3001"))
    except ImportError:
        logger.info("The 'langfuse' package is not installed; tracing is off. "
                    "This is not an error — install it to enable tracing.")
        _client = None
    except Exception as e:
        logger.warning("Could not initialise Langfuse (%s); tracing is off.", e)
        _client = None
    return _client


def tracing_enabled() -> bool:
    return _resolve_client() is not None


class _NullSpan:
    """
    Stands in for a span when tracing is off.

    Accepts and discards everything, so call sites need no conditionals. A caller that writes
    `s.update(output=expensive())` still pays for `expensive()` — compute inside the guard where
    that matters.
    """

    def update(self, **_kwargs) -> None:
        return None

    def score(self, *_args, **_kwargs) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


_NULL = _NullSpan()


@contextmanager
def _observe(label: str, *factories):
    """
    Enters the SDK's context managers with **setup and teardown guarded, and the body not**.

    Each factory takes the resolved client and returns a context manager; they are entered in
    order and the value of the last one is yielded. If any of them fails, those already entered
    are unwound and a no-op span is yielded instead, so the traced code still runs.

    The `yield` sits deliberately outside the try/except. An earlier version wrapped it too, which
    meant an exception raised by the *traced code* was thrown back into this generator, caught
    here, and — because the generator then yielded a second time — replaced by
    `RuntimeError: generator didn't stop after throw()`. The tracer destroyed the very error it
    existed to record: an indexing failure surfaced as "failed: generator didn't stop after
    throw()" rather than its cause. Errors from the body must pass through untouched; only the
    tracer's own calls are ours to swallow.
    """
    client = _resolve_client()
    if client is None:
        yield _NULL
        return

    stack = ExitStack()
    try:
        entered = _NULL
        for factory in factories:
            entered = stack.enter_context(factory(client))
    except Exception as e:
        logger.debug("%s failed to start: %s", label, e)
        _close(stack, label, (None, None, None))
        yield _NULL
        return

    try:
        yield entered
    except BaseException:
        _close(stack, label, sys.exc_info())
        raise
    _close(stack, label, (None, None, None))


def _close(stack: ExitStack, label: str, exc_info) -> None:
    """
    Unwinds a span's context managers. A failure to close is swallowed — and the return value is
    ignored, so a tracer cannot suppress an exception from the code it was observing.
    """
    try:
        stack.__exit__(*exc_info)
    except Exception as e:
        logger.debug("%s failed to close: %s", label, e)


@contextmanager
def start_trace_sync(name: str, *, session_id: Optional[str] = None,
                     user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[list] = None):
    """
    Opens a root trace from synchronous code.

    Indexing needs this: it is CPU-bound, runs in a worker thread via `run_in_executor`, and
    `run_in_executor` — unlike `asyncio.to_thread` — does not copy the caller's context, so an
    ambient trace opened on the event loop would not reach it. A background job therefore opens
    its own root rather than inheriting one.
    """
    def _attributes(_client):
        from langfuse import propagate_attributes
        return propagate_attributes(session_id=session_id, user_id=user_id,
                                    metadata=metadata, tags=tags)

    def _root(client):
        return client.start_as_current_observation(name=name, as_type="span")

    with _observe(f"Trace '{name}'", _attributes, _root) as root:
        yield root


@asynccontextmanager
async def start_trace(name: str, *, session_id: Optional[str] = None,
                      user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                      tags: Optional[list] = None):
    """
    Opens a root trace. `session_id` groups a conversation's queries into one Langfuse session.
    """
    with start_trace_sync(name, session_id=session_id, user_id=user_id,
                          metadata=metadata, tags=tags) as root:
        yield root


@contextmanager
def span(name: str, *, input: Any = None, metadata: Optional[Dict[str, Any]] = None):
    """A child span. Yields a no-op object when tracing is off, so call sites stay unconditional."""
    def _factory(client):
        return client.start_as_current_observation(
            name=name, as_type="span", input=input, metadata=metadata
        )

    with _observe(f"Span '{name}'", _factory) as s:
        yield s


@contextmanager
def generation(name: str, *, model: Optional[str] = None, input: Any = None,
               metadata: Optional[Dict[str, Any]] = None,
               model_parameters: Optional[Dict[str, Any]] = None,
               prompt: Any = None):
    """
    An LLM call. Typed as a generation so Langfuse costs it from the model and token counts.

    There are **two** of these per turn — the answer, and the conversation summariser. Tracing only
    the first understates cost by an unknown amount.

    `prompt` is the managed prompt object this call used, when there is one. It links the
    generation to a specific prompt **version**, which is what lets the dashboard group scores by
    version — the difference between being able to roll a prompt back and being able to tell that
    it needed rolling back.
    """
    def _factory(client):
        kwargs = {
            "name": name, "as_type": "generation", "input": input, "metadata": metadata,
            "model": model, "model_parameters": model_parameters,
        }
        # Passed only when present: an older SDK without the parameter would reject it, and a
        # missing link is a smaller loss than a failed generation.
        if prompt is not None:
            kwargs["prompt"] = prompt
        return client.start_as_current_observation(**kwargs)

    with _observe(f"Generation '{name}'", _factory) as g:
        yield g


def set_trace_io(*, input: Any = None, output: Any = None) -> None:
    """Records the question and the answer on the root trace, so the list view is readable."""
    client = _resolve_client()
    if client is None:
        return
    try:
        client.set_current_trace_io(input=input, output=output)
    except Exception as e:
        logger.debug("Could not set trace io: %s", e)


def score(name: str, value, *, comment: Optional[str] = None,
          data_type: Optional[str] = None) -> None:
    """
    Attaches a score to the current trace.

    Guarded like everything else: a scorer that raises must not fail the query it was scoring.
    """
    client = _resolve_client()
    if client is None:
        return
    try:
        kwargs = {"name": name, "value": value}
        if comment is not None:
            kwargs["comment"] = comment
        if data_type is not None:
            kwargs["data_type"] = data_type
        client.score_current_trace(**kwargs)
    except Exception as e:
        logger.debug("Could not record score '%s': %s", name, e)


def current_trace_id() -> Optional[str]:
    """
    The id of the trace being recorded right now, or None.

    Stored with the assistant turn so that feedback arriving minutes later can still be attached
    to the trace that produced the answer. Without it, a thumbs-down is a number with nothing
    behind it — the point of the score is being able to open the trace it belongs to and see the
    retrieval that caused it.
    """
    client = _resolve_client()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception as e:
        logger.debug("Could not read the current trace id: %s", e)
        return None


def score_trace(trace_id: str, name: str, value, *, comment: Optional[str] = None,
                data_type: Optional[str] = None) -> bool:
    """
    Scores a trace that has already finished, addressed by id.

    `score()` attaches to whatever trace is currently open, which is no use for user feedback: by
    the time someone clicks a thumb, the request that produced the answer is long over. Returns
    whether the score was actually recorded, so the endpoint can tell the caller that feedback was
    stored rather than silently accepted — tracing being off is a normal state here, not an error.
    """
    client = _resolve_client()
    if client is None or not trace_id:
        return False
    try:
        kwargs = {"name": name, "value": value, "trace_id": trace_id}
        if comment is not None:
            kwargs["comment"] = comment
        if data_type is not None:
            kwargs["data_type"] = data_type
        client.create_score(**kwargs)
        return True
    except Exception as e:
        logger.debug("Could not score trace '%s' with '%s': %s", trace_id, name, e)
        return False


def flush() -> None:
    """Forces a send. Used by scripts that exit before the background flush would run."""
    client = _resolve_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.debug("Could not flush traces: %s", e)


def shutdown_tracing() -> None:
    """
    Flushes and closes on application shutdown.

    Waits rather than abandons, for the same reason the indexing job runner waits: a queued trace
    that is dropped at exit is a trace of exactly the shutdown you wanted to see.
    """
    client = _resolve_client()
    if client is None:
        return
    try:
        client.shutdown()
        logger.info("Langfuse client shut down.")
    except Exception as e:
        logger.warning("Error shutting down Langfuse: %s", e)
