"""
Tracing and deterministic scoring.

Two properties matter more than the metrics themselves:

1. **Tracing can never fail a query.** It is developer-facing; an unreachable dashboard must not
   cost a user their answer. Every call into the SDK is guarded, and these tests prove the guards
   hold when the SDK is absent, misconfigured, or raising.
2. **Scores are honest about what they cannot measure.** A metric that does not apply is omitted
   rather than defaulted, so an average over the dashboard is an average over queries the metric
   meant something for.
"""
import os

import pytest

from app.observability import scorers
from app.observability import tracing


CHUNKS = [
    {"file_path": "svc/discount.py", "symbol": "DiscountService.calculate_discount",
     "start_line": 15, "end_line": 33},
    {"file_path": "svc/discount.py", "symbol": "DiscountService", "start_line": 3, "end_line": 37},
    {"file_path": "api/checkout.py", "symbol": "process_checkout", "start_line": 25, "end_line": 70},
    {"file_path": "tests/test_discount.py", "symbol": "test_welcome10_coupon",
     "start_line": 4, "end_line": 8},
]
CALLERS = [
    {"caller_symbol": "process_checkout", "file_path": "api/checkout.py"},
    {"caller_symbol": "test_welcome10_coupon", "file_path": "tests/test_discount.py"},
]


# ------------------------------------------------------------------ tracing never breaks anything

def test_tracing_is_off_without_keys(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_init_attempted", False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.tracing_enabled() is False


def test_spans_are_usable_when_tracing_is_off(monkeypatch):
    """Call sites must not need `if tracing_enabled()` around every span."""
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_init_attempted", True)

    with tracing.span("anything") as s:
        s.update(output={"x": 1}, metadata={"y": 2})
    with tracing.generation("llm", model="m") as g:
        g.update(output="text", usage_details={"input": 1, "output": 2})
    tracing.score("some_score", 0.5)
    tracing.set_trace_io(input="q", output="a")
    tracing.flush()


async def test_a_trace_can_be_opened_when_tracing_is_off(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_init_attempted", True)
    async with tracing.start_trace("query", session_id="conv_1") as t:
        t.update(output="ok")


async def test_an_exploding_client_does_not_propagate(monkeypatch):
    """
    The guarantee this module exists for: a broken tracer degrades tracing, nothing else.
    """
    class Exploding:
        def start_as_current_observation(self, **_kw):
            raise RuntimeError("langfuse is down")

        def score_current_trace(self, **_kw):
            raise RuntimeError("langfuse is down")

        def set_current_trace_io(self, **_kw):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_client", Exploding())
    monkeypatch.setattr(tracing, "_init_attempted", True)

    with tracing.span("boom") as s:
        s.update(output="still fine")
    async with tracing.start_trace("boom") as t:
        t.update(output="still fine")
    tracing.score("boom", 1.0)
    tracing.set_trace_io(input="q")


class _RecordingSpan:
    def __init__(self, name, closed):
        self.name = name
        self._closed = closed

    def update(self, **_kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed.append((self.name, exc[0]))
        return False


class _WorkingClient:
    """A client that behaves, so the tests below exercise the success path's unwinding."""

    def __init__(self):
        self.closed = []

    def start_as_current_observation(self, *, name, **_kwargs):
        return _RecordingSpan(name, self.closed)


@pytest.fixture
def working_client(monkeypatch):
    client = _WorkingClient()
    monkeypatch.setattr(tracing, "_client", client)
    monkeypatch.setattr(tracing, "_init_attempted", True)
    return client


def test_an_error_inside_a_span_reaches_the_caller_unchanged(working_client):
    """
    The tracer must never replace the error it is observing.

    With the `yield` inside the guard, contextlib threw the body's exception back into the
    generator, the guard caught it, and the second yield turned it into
    `RuntimeError: generator didn't stop after throw()`. Indexing reports `f"failed: {e}"`, so
    a real failure would have been recorded as that RuntimeError instead of its cause.
    """
    with pytest.raises(ValueError, match="the real error"):
        with tracing.span("work"):
            raise ValueError("the real error")


def test_a_span_is_closed_with_the_exception_that_passed_through(working_client):
    """Passing the error through is not the same as skipping teardown — the span must still end."""
    with pytest.raises(ValueError):
        with tracing.span("work"):
            raise ValueError("boom")

    assert [name for name, _ in working_client.closed] == ["work"]
    assert working_client.closed[0][1] is ValueError


def test_a_span_is_closed_on_the_success_path(working_client):
    with tracing.span("work") as s:
        s.update(output="fine")

    assert working_client.closed == [("work", None)]


async def test_an_error_inside_a_trace_reaches_the_caller_unchanged(working_client, monkeypatch):
    monkeypatch.setattr(tracing, "_resolve_client", lambda: working_client)

    with pytest.raises(ValueError, match="the real error"):
        async with tracing.start_trace("query", session_id="conv_1"):
            raise ValueError("the real error")


def test_a_failed_trace_setup_still_runs_the_body(monkeypatch):
    """
    Guarding setup is the other half: Langfuse being down degrades tracing, not indexing.
    """
    class Exploding:
        def start_as_current_observation(self, **_kw):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_client", Exploding())
    monkeypatch.setattr(tracing, "_init_attempted", True)

    ran = []
    with tracing.start_trace_sync("index") as t:
        t.update(output="still fine")
        ran.append(True)

    assert ran == [True]


def test_importing_tracing_alone_loads_the_env_file():
    """
    Keys are read with `os.getenv`, and `backend/.env` reaches the environment only because
    `app.core.config` loads it at import. Nothing else in this module's chain pulls config in, so
    without an explicit import whether tracing worked depended on import *order*: the application
    imports config early and traced fine, while a standalone script importing only this module
    found no keys and reported "tracing is off" — with a log line that looked like the keys were
    genuinely unset.

    Run in a **subprocess** because it cannot be observed in this one: `conftest.py` imports
    config long before any test runs, so in-process `sys.modules` always contains it and the
    check passes whether or not the import exists. A clean interpreter is the only place the
    dependency is visible.
    """
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c",
         "import app.observability.tracing, sys; "
         "print('app.core.config' in sys.modules)"],
        cwd=str(backend), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip() == "True", (
        "importing tracing must pull in app.core.config so backend/.env is loaded "
        "regardless of import order"
    )


def test_initialisation_is_attempted_once(monkeypatch):
    """A misconfigured deployment logs once at startup, not once per query."""
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_init_attempted", False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    tracing._resolve_client()
    assert tracing._init_attempted is True
    tracing._resolve_client()


# ------------------------------------------------------------------ symbol matching

def test_a_symbol_inside_a_longer_name_is_not_a_mention():
    """
    The same trap the intent classifier fell into: substring matching read 'unchanged' as
    'change'. Here it would read a test's name as a mention of the function it tests.
    """
    assert scorers._mentions("see test_calculate_discount_rounding", "calculate_discount") is False
    assert scorers._mentions("calls calculate_discount()", "calculate_discount") is True


def test_a_qualified_symbol_is_found_by_its_tail():
    """Models write `calculate_discount`; the graph stores `DiscountService.calculate_discount`."""
    assert scorers._mentions("`calculate_discount` is called here",
                             "DiscountService.calculate_discount") is True


def test_backticks_and_parentheses_do_not_hide_a_mention():
    for form in ("`process_checkout`", "process_checkout()", "**process_checkout**",
                 "| `process_checkout` |"):
        assert scorers._mentions(f"It is called by {form} there.", "process_checkout"), form


# ------------------------------------------------------------------ graph as oracle

def test_graph_recall_counts_named_callers():
    answer = "Called by `process_checkout` and by `test_welcome10_coupon`."
    assert scorers.graph_recall(answer, CALLERS) == 1.0

    partial = "Called by `process_checkout`."
    assert scorers.graph_recall(partial, CALLERS) == 0.5


def test_graph_recall_is_none_when_there_is_nothing_to_find():
    """
    Scoring 1.0 for "the graph knew of no callers" would quietly inflate the average with queries
    the metric said nothing about.
    """
    assert scorers.graph_recall("anything", []) is None


def test_graph_precision_catches_an_invented_caller():
    answer = "Called by `process_checkout` and by `DiscountService`."
    # DiscountService is a real symbol in the repository but not a caller.
    assert scorers.graph_precision(answer, CALLERS, CHUNKS) == 0.5


def test_graph_scores_only_apply_to_graph_adjudicated_intents():
    """For a location or feature question the right answer is not a caller list."""
    answer = "It lives in svc/discount.py."
    for intent in ("CODE_LOCATION", "FEATURE_EXPLANATION"):
        got = scorers.collect(answer=answer, sources=[], all_chunks=CHUNKS,
                              intent=intent, graph_callers=CALLERS)
        assert "graph_recall" not in got, intent

    got = scorers.collect(answer=answer, sources=[], all_chunks=CHUNKS,
                          intent="DEPENDENCY_ANALYSIS", graph_callers=CALLERS)
    assert "graph_recall" in got


# ------------------------------------------------------------------ citations

def test_citation_validity_checks_the_file_exists():
    good = [{"file_path": "svc/discount.py", "start_line": 15}]
    bad = [{"file_path": "svc/invented.py", "start_line": 1}]
    assert scorers.citation_validity(good, CHUNKS) == 1.0
    assert scorers.citation_validity(bad, CHUNKS) == 0.0
    assert scorers.citation_validity(good + bad, CHUNKS) == 0.5


def test_citation_line_validity_catches_a_range_past_the_end_of_a_file():
    """What a stale citation looks like after a sync: right file, wrong lines."""
    in_range = [{"file_path": "svc/discount.py", "start_line": 20}]
    past_end = [{"file_path": "svc/discount.py", "start_line": 5000}]
    assert scorers.citation_line_validity(in_range, CHUNKS) == 1.0
    assert scorers.citation_line_validity(past_end, CHUNKS) == 0.0


def test_citation_scores_are_none_with_no_citations():
    assert scorers.citation_validity([], CHUNKS) is None
    assert scorers.citation_line_validity([], CHUNKS) is None


def test_path_grounding_flags_a_path_the_model_never_saw():
    sources = [{"file_path": "svc/discount.py"}]
    assert scorers.path_grounding("See `svc/discount.py`.", sources) == 1.0
    assert scorers.path_grounding("See `svc/imaginary.py`.", sources) == 0.0


def test_a_path_from_the_graph_facts_is_grounded_even_if_it_was_not_retrieved():
    """
    Found live, on a verified-correct answer that scored 0.38.

    The prompt shows the model two things, and rule A1 tells it to report the complete caller
    list from the *facts* even when the snippets show fewer. Scoring against the snippets alone
    marked every caller retrieval did not happen to return as invented — so the metric fired
    hardest exactly when the model did what it was told.
    """
    sources = [{"file_path": "services/module_1.py"}]
    callers = [{"caller_symbol": "operation_2", "file_path": "services/module_2.py"},
               {"caller_symbol": "handle_request", "file_path": "api/handlers.py"}]
    answer = ("Called from `services/module_1.py`, `services/module_2.py` "
              "and `api/handlers.py`.")

    assert scorers.path_grounding(answer, sources) < 0.4, "the old behaviour, for contrast"
    assert scorers.path_grounding(answer, sources, callers) == 1.0


def test_an_invented_path_is_still_caught_when_graph_facts_are_present():
    """Widening what counts as grounded must not stop it catching a genuine invention."""
    sources = [{"file_path": "services/module_1.py"}]
    callers = [{"caller_symbol": "operation_2", "file_path": "services/module_2.py"}]
    answer = "Called from `services/module_1.py` and `services/fabricated.py`."

    assert scorers.path_grounding(answer, sources, callers) == 0.5


def test_retrieval_hit_reports_whether_the_subject_was_retrieved():
    sources = [{"symbol": "DiscountService.calculate_discount"}]
    assert scorers.retrieval_hit("calculate_discount", sources) is True
    assert scorers.retrieval_hit("process_checkout", sources) is False
    assert scorers.retrieval_hit("", sources) is None


def test_prompt_budget_used_is_a_fraction():
    assert scorers.prompt_budget_used(10000, 20000) == 0.5
    assert scorers.prompt_budget_used(0, 0) is None


# ------------------------------------------------------------------ collect

def test_collect_omits_metrics_that_do_not_apply():
    got = scorers.collect(answer="", sources=[], all_chunks=CHUNKS,
                          intent="CODE_LOCATION", target_symbol="")
    assert "citation_validity" not in got
    assert "retrieval_hit" not in got
    # Booleans always apply — they describe the turn, not the answer's content.
    assert got["degraded"] is False
    assert got["facts_truncated"] is False


def test_collect_produces_the_expected_shape_for_a_dependency_answer():
    answer = "Called by `process_checkout` and `test_welcome10_coupon`, see `svc/discount.py`."
    got = scorers.collect(
        answer=answer,
        sources=[{"file_path": "svc/discount.py", "symbol": "DiscountService.calculate_discount",
                  "start_line": 15}],
        all_chunks=CHUNKS,
        intent="DEPENDENCY_ANALYSIS",
        target_symbol="calculate_discount",
        graph_callers=CALLERS,
        prompt_chars=10000,
        budget_chars=20000,
    )
    assert got["graph_recall"] == 1.0
    assert got["citation_validity"] == 1.0
    assert got["retrieval_hit"] is True
    assert got["prompt_budget_used"] == 0.5
    assert got["path_grounding"] == 1.0


def test_collect_never_raises_on_malformed_input():
    """Scoring runs after a real answer was produced; it must not be able to discard one."""
    for bad in ({"answer": None}, {"sources": None}, {"all_chunks": None}):
        kwargs = {"answer": "x", "sources": [], "all_chunks": CHUNKS, "intent": "CODE_LOCATION"}
        kwargs.update(bad)
        try:
            scorers.collect(**kwargs)
        except Exception as e:
            pytest.fail(f"collect raised on {bad}: {e}")


def test_graph_precision_does_not_count_the_subject_as_a_claimed_caller():
    """
    An answer to "what calls `calculate_discount`?" names `calculate_discount` throughout. It is
    the subject, not a claimed caller of itself — counting it scored a perfectly correct answer
    at 0.67 on a real query.
    """
    answer = ("`calculate_discount` is called by `process_checkout` "
              "and `test_welcome10_coupon`.")
    without_subject = scorers.graph_precision(
        answer, CALLERS, CHUNKS, target_symbol="DiscountService.calculate_discount")
    assert without_subject == 1.0

    # Without the exclusion the same correct answer is penalised.
    with_subject = scorers.graph_precision(answer, CALLERS, CHUNKS)
    assert with_subject < 1.0


def test_retrieval_scores_are_exposed_without_mutating_the_cached_chunks(temp_repo):
    """
    The chunks are shared cache objects, also used to build stored citations. A per-query score
    written onto them would leak into both.
    """
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import CodeReranker

    _, repo_id, chunks = temp_repo({
        "svc.py": "def charge(amount):\n    return amount * 2\n",
    })
    before = {k for c in chunks for k in c}

    retrieved = HybridRetriever.retrieve(
        repository_id=repo_id, query="what charges the amount?", all_chunks=chunks, top_k=3)

    assert CodeReranker.last_scores, "scores were not exposed"
    assert all("rerank" in v and "fused" in v for v in CodeReranker.last_scores.values())

    after = {k for c in chunks for k in c}
    assert after == before, "retrieval wrote a score onto the shared chunk objects"
    assert all("rerank" not in c for c in retrieved)


# ------------------------------------------------------------------ the indexing trace

class _TreeSpan:
    """Records its own name, payloads and children, so a test can assert on the shape."""

    def __init__(self, name, kind, input, metadata, recorder):
        self.name = name
        self.kind = kind
        self.input = input
        self.metadata = metadata
        self.output = None
        self.children = []
        self._recorder = recorder

    def update(self, **kwargs):
        if "output" in kwargs:
            self.output = kwargs["output"]

    def __enter__(self):
        self._recorder.stack.append(self)
        return self

    def __exit__(self, *_exc):
        self._recorder.stack.pop()
        return False


class _Recorder:
    """A stand-in Langfuse client that builds the span tree in memory."""

    def __init__(self):
        self.stack = []
        self.roots = []
        # Trace-level attributes arrive through `propagate_attributes`, not on the observation,
        # which is where Langfuse wants session id, tags and trace metadata.
        self.attributes = {}

    def start_as_current_observation(self, *, name, as_type="span", input=None,
                                     metadata=None, **_kwargs):
        node = _TreeSpan(name, as_type, input, metadata, self)
        if self.stack:
            self.stack[-1].children.append(node)
        else:
            self.roots.append(node)
        return node


@pytest.fixture
def recorder(monkeypatch):
    import contextlib as _contextlib
    import langfuse as _langfuse

    rec = _Recorder()
    monkeypatch.setattr(tracing, "_client", rec)
    monkeypatch.setattr(tracing, "_init_attempted", True)

    @_contextlib.contextmanager
    def _capture(**kwargs):
        rec.attributes = kwargs
        yield

    monkeypatch.setattr(_langfuse, "propagate_attributes", _capture)
    return rec


INDEX_FILES = {
    "svc/discount.py": "class DiscountService:\n"
                       "    def calculate_discount(self, total):\n"
                       "        return total * 0.1\n",
    "api/checkout.py": "from svc.discount import DiscountService\n\n"
                       "def process_checkout(cart):\n"
                       "    return DiscountService().calculate_discount(cart)\n",
}


def test_indexing_emits_one_trace_with_the_six_stages(recorder, temp_repo):
    """
    Indexing is a background job nobody watches, which is exactly why it needs a trace: ~89% of
    the system's compute is here, and a failure only surfaces later as a 409.
    """
    temp_repo(INDEX_FILES)

    assert len(recorder.roots) == 1
    root = recorder.roots[0]
    assert root.name == "index"
    assert [child.name for child in root.children] == [
        "load_previous_index", "discover_parse_chunk", "prune_stale",
        "embed", "graph", "persist",
    ]
    assert root.output["status"] == "ready"
    assert root.output["files"] == 2


def test_the_index_trace_carries_the_job_it_belongs_to(recorder, temp_repo, tmp_path):
    """
    The trace opens its own root because `run_in_executor` does not copy the caller's context,
    so the job id has to travel as metadata or the trace cannot be tied to the job being polled.
    """
    from app.api.repositories import index_repository_folder

    root_dir = tmp_path / "jobrepo"
    (root_dir / "svc").mkdir(parents=True)
    (root_dir / "svc" / "a.py").write_text("def f():\n    return 1\n")
    index_repository_folder(str(root_dir), "job_trace_repo", "job-trace",
                            job_id="job-123", trigger="github_sync")

    metadata = recorder.attributes["metadata"]
    assert metadata["job_id"] == "job-123"
    assert metadata["trigger"] == "github_sync"
    assert metadata["repository_id"] == "job_trace_repo"
    assert recorder.attributes["tags"] == ["index", "github_sync"]


def test_the_reuse_counts_are_on_the_trace(recorder, temp_repo):
    """
    "1 re-parsed, 2 reused" is the claim the incremental path rests on. It was only ever a log
    line; on the span it is the first thing visible when a sync looks too slow.
    """
    _, repo_id, _ = temp_repo(INDEX_FILES)
    recorder.roots.clear()

    from app.api.repositories import index_repository_folder
    import app.api.repositories as repos

    root_dir, _, _ = temp_repo(INDEX_FILES, repo_id=repo_id + "_again")
    recorder.roots.clear()
    # Re-index the same source, unchanged: everything should be reused, nothing re-parsed.
    index_repository_folder(root_dir, repo_id + "_again", "again")

    stage = {c.name: c for c in recorder.roots[0].children}["discover_parse_chunk"]
    assert stage.output["reused_files"] == 2
    assert stage.output["parsed_files"] == 0
    assert repos.indexing_status[repo_id + "_again"] == "ready"


def test_a_failing_index_is_recorded_on_the_trace_with_its_real_error(recorder, monkeypatch,
                                                                     tmp_path):
    """
    The failure path is the reason the span guards had to be fixed first: with the body inside
    the guard, this error arrived as "generator didn't stop after throw()".
    """
    import app.api.repositories as repos

    def explode(*_a, **_kw):
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(repos.chroma_store, "add_chunks", explode)

    root_dir = tmp_path / "failrepo"
    root_dir.mkdir()
    (root_dir / "a.py").write_text("def f():\n    return 1\n")

    with pytest.raises(RuntimeError, match="chroma is down"):
        repos.index_repository_folder(str(root_dir), "fail_repo", "fail")

    root = recorder.roots[0]
    assert root.output["status"] == "failed"
    assert "chroma is down" in root.output["error"]
    assert repos.indexing_status["fail_repo"] == "failed: chroma is down"


# ------------------------------------------------------------------ the acquisition trace

def _stage_names(root):
    return [child.name for child in root.children]


def test_acquiring_a_repository_traces_clone_limits_index_and_cleanup(recorder, monkeypatch,
                                                                     tmp_path):
    """
    One trace per import job, covering acquisition and indexing together.

    The index trace nests inside rather than standing alone: both run on the same worker thread,
    so the ambient context carries, and "the clone took 20s and the embed took 40s" is one
    picture rather than two traces nobody joins up.
    """
    import shutil as _shutil
    import app.api.github as gh

    source = tmp_path / "src"
    (source / "svc").mkdir(parents=True)
    (source / "svc" / "discount.py").write_text(
        "class DiscountService:\n    def calculate_discount(self, total):\n        return total\n"
    )

    def fake_clone(url, branch, dest, **_kwargs):
        _shutil.copytree(str(source), dest)
        return "c" * 40

    monkeypatch.setattr(gh, "clone_repository", fake_clone)

    result = gh._clone_and_index("acq_repo", "https://github.com/o/r.git", "main", "o/r@main")

    assert result["commit_sha"] == "c" * 40
    root = recorder.roots[0]
    assert root.name == "acquire"
    assert _stage_names(root) == ["clone", "enforce_limits", "index", "cleanup"]
    assert root.output["status"] == "ok"

    clone_span, limits_span, index_span, cleanup_span = root.children
    assert clone_span.output["commit_sha"] == "c" * 40
    assert limits_span.output["file_count"] == 1
    assert limits_span.output["total_bytes"] > 0
    assert _stage_names(index_span) == [
        "load_previous_index", "discover_parse_chunk", "prune_stale",
        "embed", "graph", "persist",
    ]
    assert cleanup_span.output["removed"] is True


def test_the_working_tree_is_removed_and_the_trace_says_so_when_indexing_fails(recorder,
                                                                              monkeypatch,
                                                                              tmp_path):
    """
    Cleanup is in a `finally`, so it must still run — and still be traced — when the job fails.
    Leaving a clone behind is how the disk fills; a span is what makes that visible.
    """
    import shutil as _shutil
    import app.api.github as gh

    source = tmp_path / "src2"
    source.mkdir()
    (source / "a.py").write_text("def f():\n    return 1\n")

    def fake_clone(url, branch, dest, **_kwargs):
        _shutil.copytree(str(source), dest)
        return "d" * 40

    def explode(*_a, **_kw):
        raise RuntimeError("indexing blew up")

    monkeypatch.setattr(gh, "clone_repository", fake_clone)
    monkeypatch.setattr(gh, "index_repository_folder", explode)

    destination = gh._clone_dir("acq_fail_repo")
    with pytest.raises(RuntimeError, match="indexing blew up"):
        gh._clone_and_index("acq_fail_repo", "https://github.com/o/r.git", "main", "o/r@main")

    root = recorder.roots[0]
    assert root.output["status"] == "failed"
    assert "indexing blew up" in root.output["error"]
    assert "cleanup" in _stage_names(root)
    assert not os.path.isdir(destination), "a failed import must not leave a clone on disk"


def test_a_failed_clone_is_visible_on_the_trace(recorder, monkeypatch):
    """A clone that never succeeds still produces a trace — otherwise the failure is invisible."""
    import app.api.github as gh
    from app.ingestion.git_source import CloneFailed

    def failing_clone(*_a, **_kw):
        raise CloneFailed("repository not found")

    monkeypatch.setattr(gh, "clone_repository", failing_clone)

    with pytest.raises(CloneFailed):
        gh._clone_and_index("acq_gone", "https://github.com/o/gone.git", "main", "o/gone@main")

    root = recorder.roots[0]
    assert root.name == "acquire"
    assert "repository not found" in root.output["error"]
    # The clone span is opened and the cleanup still runs; limits and index never do.
    assert _stage_names(root) == ["clone", "cleanup"]


def test_the_sync_trace_carries_the_reuse_counts(recorder, monkeypatch, tmp_path):
    """
    "Why was this sync slow?" is answered by reused vs re-parsed, so the sync root carries them.
    """
    import shutil as _shutil
    import app.api.github as gh

    source = tmp_path / "src3"
    source.mkdir()
    (source / "a.py").write_text("def f():\n    return 1\n")
    (source / "b.py").write_text("def g():\n    return 2\n")

    def fake_clone(url, branch, dest, **_kwargs):
        _shutil.copytree(str(source), dest)
        return "e" * 40

    monkeypatch.setattr(gh, "clone_repository", fake_clone)

    gh._clone_and_reindex("sync_repo", "https://github.com/o/r.git", "main", "o/r@main", False)
    recorder.roots.clear()
    # Same bytes, fresh directory: the manifest keys on content hash, so nothing is re-parsed.
    gh._clone_and_reindex("sync_repo", "https://github.com/o/r.git", "main", "o/r@main", False)

    root = recorder.roots[0]
    assert root.output["reused_files"] == 2
    assert root.output["parsed_files"] == 0


# ------------------------------------------------------------------ the snippet trace

SNIPPET_FILES = {
    "billing.py": (
        "def add_item(cart, item):\n"
        "    cart.append(item)\n"
        "    return cart\n"
        "\n"
        "\n"
        "def calculate_total(items):\n"
        "    subtotal = sum(i.price for i in items)\n"
        "    return subtotal\n"
    )
}


def _symbol_chunk(chunks, symbol):
    """The indexed chunk for a symbol, so tests do not hard-code the chunker's line numbers."""
    for chunk in chunks:
        if chunk.get("symbol") == symbol and chunk.get("file_path") == "billing.py":
            return chunk
    raise AssertionError(f"'{symbol}' was not indexed")


async def test_resolving_a_citation_is_traced(recorder, temp_repo):
    """
    Citation expands happen long after the answer that cited them, so there is no query trace to
    attach to — this is a request of its own and gets its own root.
    """
    import app.api.repositories as repos

    _, repo_id, chunks = temp_repo(SNIPPET_FILES)
    target = _symbol_chunk(chunks, "calculate_total")
    recorder.roots.clear()

    payload = await repos.get_snippet(repo_id, "billing.py",
                                      start_line=target["start_line"],
                                      end_line=target["end_line"])

    assert payload["symbol"] == "calculate_total"
    root = recorder.roots[0]
    assert root.name == "snippet"
    assert _stage_names(root) == ["load_chunks", "resolve_pointer"]
    assert root.output["match"] == "exact"
    assert root.output["stale"] is False

    resolve = root.children[1]
    assert resolve.output["match"] == "exact"
    assert resolve.output["line_offset"] == 0
    assert resolve.output["symbol"] == "calculate_total"


async def test_a_pointer_landing_inside_a_symbol_is_traced_as_enclosing(recorder, temp_repo):
    """
    `match` is the reason this endpoint is worth tracing: it is the difference between showing
    the reader the code that was cited and showing them something near it.
    """
    import app.api.repositories as repos

    _, repo_id, chunks = temp_repo(SNIPPET_FILES)
    target = _symbol_chunk(chunks, "calculate_total")
    recorder.roots.clear()

    payload = await repos.get_snippet(repo_id, "billing.py",
                                      start_line=target["start_line"] + 1,
                                      end_line=target["end_line"])

    assert payload["symbol"] == "calculate_total"
    resolve = recorder.roots[0].children[1]
    assert resolve.output["match"] == "enclosing"
    assert resolve.output["line_offset"] == -1


async def test_a_stale_citation_records_both_shas(recorder, temp_repo, monkeypatch):
    """The `at_sha` check is what stops the viewer confidently showing the wrong code."""
    import app.api.repositories as repos

    _, repo_id, chunks = temp_repo(SNIPPET_FILES)
    target = _symbol_chunk(chunks, "calculate_total")

    class _Col:
        async def find_one(self, *_a, **_kw):
            return {"repository_id": repo_id, "commit_sha": "b" * 40}

    monkeypatch.setattr(repos.db_client, "get_collection", lambda _name: _Col())
    recorder.roots.clear()

    payload = await repos.get_snippet(repo_id, "billing.py",
                                      start_line=target["start_line"],
                                      end_line=target["end_line"],
                                      at_sha="a" * 40)

    assert payload["stale"] is True
    assert payload["moved"] is False

    root = recorder.roots[0]
    assert _stage_names(root) == ["load_chunks", "resolve_pointer", "staleness_check"]
    check = root.children[2]
    assert check.output["stale"] is True
    assert check.output["indexed_sha"] == "b" * 40
    assert check.output["comparable"] is True
    assert root.output["stale"] is True


async def test_a_repository_with_no_recorded_sha_is_not_called_fresh(recorder, temp_repo,
                                                                     monkeypatch):
    """
    A zip upload has no commit sha, so staleness is unknowable — which is not the same as
    checked-and-fresh, and the trace has to keep the two apart.
    """
    import app.api.repositories as repos

    _, repo_id, chunks = temp_repo(SNIPPET_FILES)
    target = _symbol_chunk(chunks, "calculate_total")

    class _Col:
        async def find_one(self, *_a, **_kw):
            return {"repository_id": repo_id}

    monkeypatch.setattr(repos.db_client, "get_collection", lambda _name: _Col())
    recorder.roots.clear()

    payload = await repos.get_snippet(repo_id, "billing.py",
                                      start_line=target["start_line"],
                                      end_line=target["end_line"],
                                      at_sha="a" * 40)

    assert payload["stale"] is False
    check = recorder.roots[0].children[2]
    assert check.output["comparable"] is False
    assert check.output["stale"] is False


async def test_an_unindexed_repository_is_visible_on_the_snippet_trace(recorder):
    """The 409 and the 404 have different fixes, so the trace has to tell them apart."""
    from fastapi import HTTPException
    import app.api.repositories as repos

    with pytest.raises(HTTPException) as caught:
        await repos.get_snippet("never_indexed_repo", "billing.py", start_line=1, end_line=2)

    assert caught.value.status_code == 409
    root = recorder.roots[0]
    assert _stage_names(root) == ["load_chunks"]
    assert root.children[0].output["indexed"] is False


async def test_a_file_outside_the_repository_is_traced_as_such(recorder, temp_repo):
    import app.api.repositories as repos
    from fastapi import HTTPException

    _, repo_id, _ = temp_repo(SNIPPET_FILES)
    recorder.roots.clear()

    with pytest.raises(HTTPException) as caught:
        await repos.get_snippet(repo_id, "nope.py", start_line=1, end_line=2)

    assert caught.value.status_code == 404
    resolve = recorder.roots[0].children[1]
    assert resolve.output["match"] == "no_such_file"
    assert resolve.output["candidates"] == 0
