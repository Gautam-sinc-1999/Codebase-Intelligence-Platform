"""
Running an evaluation dataset.

The runner is a measuring instrument, so the tests are mostly about it measuring *honestly*:

- a wrong answer must score 0, not be quietly skipped;
- one broken item must not abandon the other twenty-two;
- the fixture must be removed afterwards, because an evaluation run that leaves a repository in
  Neo4j and ChromaDB is one that slowly fills them.

Everything here runs with no Langfuse reachable. The scoring functions are deliberately free of
SDK objects so they can be tested exactly as the script uses them.
"""
import pytest

from app.observability import datasets as ds
from app.observability import runner


# ------------------------------------------------------------------ executing an item

def test_an_intent_item_runs_the_classifier():
    output = runner.execute({"query": "Where is authentication implemented?"},
                            {"check": "intent"})
    assert output["intent"] == "CODE_LOCATION"
    assert output["keyword_score"] > 0


def test_a_trap_item_reports_the_keyword_score():
    output = runner.execute({"query": "Which parts are unchanged since last release?"},
                            {"check": "no_fragment_match"})
    assert output["keyword_score"] == 0


def test_an_unknown_check_is_refused_rather_than_guessed():
    """
    Guessing from the input shape would be wrong: the two intent checks take identical inputs and
    differ only in what counts as correct.
    """
    with pytest.raises(ValueError, match="No execution path"):
        runner.execute({"query": "anything"}, {"check": "invented"})


def test_a_graph_item_without_a_fixture_is_refused():
    with pytest.raises(ValueError, match="GraphFixture"):
        runner.execute({"query": "Who calls record_audit?", "symbol": "record_audit"},
                       {"check": "graph_recall"})


# ------------------------------------------------------------------ scoring

def test_a_correct_intent_scores_one():
    got = runner.evaluate({"intent": "CODE_LOCATION"}, {"intent": "CODE_LOCATION"},
                          {"check": "intent"})
    assert got == [{"name": "intent_correct", "value": 1.0, "comment": None}]


def test_a_wrong_intent_scores_zero_and_says_what_it_expected():
    got = runner.evaluate({"intent": "CHANGE_IMPACT"}, {"intent": "CODE_LOCATION"},
                          {"check": "intent"})
    assert got[0]["value"] == 0.0
    assert "expected CODE_LOCATION" in got[0]["comment"]
    assert "got CHANGE_IMPACT" in got[0]["comment"]


def test_a_fragment_match_scores_zero():
    """A non-zero keyword score is the bug, whatever intent the classifier ultimately returned."""
    got = runner.evaluate({"keyword_score": 3, "intent": "CHANGE_IMPACT"},
                          {"keyword_score": 0},
                          {"check": "no_fragment_match", "reason": "'unchanged' contains 'change'"})
    assert got[0]["value"] == 0.0
    assert "contains 'change'" in got[0]["comment"]


def test_graph_recall_is_the_fraction_of_known_callers_found():
    expected = {"callers": ["a", "b", "c", "d"]}
    got = runner.evaluate({"callers": ["a", "b"]}, expected, {"check": "graph_recall"})
    by_name = {e["name"]: e for e in got}

    assert by_name["graph_caller_recall"]["value"] == 0.5
    assert "missing: ['c', 'd']" in by_name["graph_caller_recall"]["comment"]
    assert by_name["graph_caller_precision"]["value"] == 1.0


def test_an_invented_caller_costs_precision_not_recall():
    expected = {"callers": ["a", "b"]}
    got = runner.evaluate({"callers": ["a", "b", "ghost"]}, expected, {"check": "graph_recall"})
    by_name = {e["name"]: e for e in got}

    assert by_name["graph_caller_recall"]["value"] == 1.0
    assert by_name["graph_caller_precision"]["value"] == pytest.approx(2 / 3)
    assert "invented: ['ghost']" in by_name["graph_caller_precision"]["comment"]


def test_finding_nothing_scores_zero_rather_than_dividing_by_zero():
    got = runner.evaluate({"callers": []}, {"callers": ["a"]}, {"check": "graph_recall"})
    by_name = {e["name"]: e for e in got}
    assert by_name["graph_caller_recall"]["value"] == 0.0
    assert by_name["graph_caller_precision"]["value"] == 0.0


def test_facts_mode_does_not_score_the_prose():
    """There is no answer in `facts` mode, so scoring one would be inventing a measurement."""
    got = runner.evaluate({"callers": ["a"], "mode": "facts"}, {"callers": ["a"]},
                          {"check": "graph_recall"})
    assert {e["name"] for e in got} == {"graph_caller_recall", "graph_caller_precision"}


def test_answer_mode_also_checks_the_answer_named_the_callers():
    """
    The G-01 failure exactly: the graph holds all thirteen while the answer names a handful.
    Caller recall alone would report a perfect run.
    """
    output = {
        "callers": ["alpha", "beta", "gamma", "delta"],
        "answer": "It is called by `alpha` and `beta`.",
        "mode": "answer",
        "degraded": False,
    }
    got = runner.evaluate(output, {"callers": ["alpha", "beta", "gamma", "delta"]},
                          {"check": "graph_recall"})
    by_name = {e["name"]: e for e in got}

    assert by_name["graph_caller_recall"]["value"] == 1.0
    assert by_name["answer_names_callers"]["value"] == 0.5
    assert by_name["degraded"]["value"] == 1.0


def test_a_degraded_answer_is_flagged():
    output = {"callers": ["a"], "answer": "a", "mode": "answer", "degraded": True}
    got = runner.evaluate(output, {"callers": ["a"]}, {"check": "graph_recall"})
    degraded = [e for e in got if e["name"] == "degraded"][0]
    assert degraded["value"] == 0.0
    assert "template" in degraded["comment"]


# ------------------------------------------------------------------ the run as a whole

def test_one_broken_item_does_not_abandon_the_rest():
    """
    A run that stops at item 3 of 23 says nothing about the other 20, and the usual cause is a
    fixture problem rather than a regression.
    """
    items = [
        {"id": "ok", "input": {"query": "Where is authentication implemented?"},
         "expected_output": {"intent": "CODE_LOCATION"}, "metadata": {"check": "intent"}},
        {"id": "broken", "input": {"query": "x"},
         "expected_output": {}, "metadata": {"check": "nonsense"}},
        {"id": "ok2", "input": {"query": "Who calls process_checkout?"},
         "expected_output": {"intent": "DEPENDENCY_ANALYSIS"}, "metadata": {"check": "intent"}},
    ]
    results = runner.run_items(items)

    assert [r["id"] for r in results] == ["ok", "broken", "ok2"]
    assert results[0]["passed"] is True
    assert results[1]["passed"] is False and results[1]["error"]
    assert results[2]["passed"] is True


def test_the_whole_intent_dataset_passes_today():
    """
    The datasets exist to catch a regression, so they must pass now — a suite that starts red has
    no signal left to give.
    """
    results = runner.run_items(ds.intent_items())
    summary = runner.summarise(results)

    assert summary["items"] == 21
    assert summary["failed"] == 0, summary["failures"]
    assert summary["metrics"]["intent_correct"] == 1.0
    assert summary["metrics"]["no_fragment_match"] == 1.0


def test_a_summary_reports_means_and_the_failures_behind_them():
    results = [
        {"id": "a", "query": "q1", "check": "intent", "passed": True,
         "evaluations": [{"name": "intent_correct", "value": 1.0, "comment": None}]},
        {"id": "b", "query": "q2", "check": "intent", "passed": False,
         "evaluations": [{"name": "intent_correct", "value": 0.0, "comment": "expected X"}]},
    ]
    summary = runner.summarise(results)

    assert summary["metrics"]["intent_correct"] == 0.5
    assert summary["passed"] == 1 and summary["failed"] == 1
    assert summary["failures"] == [{"query": "q2", "check": "intent", "why": "expected X"}]


def test_an_errored_item_carries_its_reason_into_the_summary():
    results = runner.run_items([
        {"id": "bad", "input": {"query": "x"}, "expected_output": {},
         "metadata": {"check": "nonsense"}},
    ])
    summary = runner.summarise(results)
    assert summary["errors"] == 1
    assert "No execution path" in summary["failures"][0]["why"]


# ------------------------------------------------------------------ the fixture

def test_the_graph_fixture_is_removed_after_a_run():
    """
    An evaluation that leaves its fixture behind fills Neo4j and ChromaDB with repositories
    nothing owns — invisible, because they have no Mongo record to appear in the sidebar.
    """
    import os

    from app.ingestion.index_store import index_store

    with runner.GraphFixture("eval_cleanup_probe") as fixture:
        repository_id = fixture.ensure()
        root = fixture._root
        assert os.path.isdir(root)
        assert index_store.exists(repository_id)

    assert not os.path.isdir(root), "the fixture's working tree survived the run"
    assert not index_store.exists(repository_id), "the fixture's index survived the run"


def test_the_fixture_is_built_once_and_reused():
    """Indexing is the expensive part, and every graph item asks about the same repository."""
    with runner.GraphFixture("eval_reuse_probe") as fixture:
        first = fixture.ensure()
        root = fixture._root
        second = fixture.ensure()
        assert first == second
        assert fixture._root == root


def test_graph_items_score_full_recall_against_the_real_graph():
    """End to end: the dataset's ground truth matched against a genuinely indexed repository."""
    with runner.GraphFixture("eval_end_to_end") as fixture:
        results = runner.run_items(ds.graph_items(), fixture=fixture)

    summary = runner.summarise(results)
    assert summary["failed"] == 0, summary["failures"]
    assert summary["metrics"]["graph_caller_recall"] == 1.0
    assert summary["metrics"]["graph_caller_precision"] == 1.0


# ------------------------------------------------------------------ answer mode is async

async def test_answer_mode_works_from_inside_a_running_event_loop():
    """
    Found live, and only live: the RAGAS harness is async, and `_answer_output` used
    `asyncio.run` — which raises "cannot be called from a running event loop" the moment the
    caller already has one. Every test until now exercised `facts` mode end to end and `answer`
    mode only through synthetic output dicts, so the one path that mattered was never run.

    This test is `async` on purpose: in a sync test the old code would have passed.
    """
    with runner.GraphFixture("eval_async_probe") as fixture:
        item = ds.graph_items()[0]
        output = await runner.aexecute(item["input"], item["metadata"],
                                       mode="answer", fixture=fixture)

    assert output["mode"] == "answer"
    assert isinstance(output["answer"], str) and output["answer"].strip()
    # The graph half must still be right regardless of what the model wrote.
    assert set(output["callers"]) == set(ds.GRAPH_EXPECTED_CALLERS)


async def test_the_sync_execute_refuses_answer_mode_rather_than_breaking():
    """A clear error beats "cannot be called from a running event loop" three frames deep."""
    with runner.GraphFixture("eval_sync_refusal") as fixture:
        item = ds.graph_items()[0]
        with pytest.raises(ValueError, match="aexecute"):
            runner.execute(item["input"], item["metadata"], mode="answer", fixture=fixture)


async def test_aexecute_still_handles_facts_mode():
    """The async form must cover both, or callers need to know which to use when."""
    output = await runner.aexecute({"query": "Where is authentication implemented?"},
                                   {"check": "intent"})
    assert output["intent"] == "CODE_LOCATION"
