"""
The evaluation datasets.

A dataset is a measuring instrument, and the failure mode that matters is not a crash — it is a
dataset that is quietly *wrong*: a mislabelled case, a duplicated item, or ground truth that has
drifted away from the fixture it describes. Every one of those makes a run report a regression
that is not there, or hide one that is.

So these tests check the labels against the real classifier and the real graph, not against a
copy of themselves.
"""
import pytest

from app.observability import datasets as ds
from app.retrieval.intent_classifier import QueryIntentClassifier as IC


# ------------------------------------------------------------------ the labels are true

@pytest.mark.parametrize("query,intent", ds.INTENT_CASES)
def test_every_intent_label_is_the_one_the_classifier_produces(query, intent):
    """
    Ground truth has to be true. A dataset whose labels disagree with correct behaviour reports a
    permanent failure, and the usual response to a permanently failing check is to stop reading it.
    """
    assert IC.classify(query)["intent"] == intent


@pytest.mark.parametrize("query,reason", ds.INTENT_TRAP_CASES)
def test_every_trap_case_really_scores_zero(query, reason):
    assert IC.classify(query)["score"] == 0, f"no longer a clean trap: {reason}"


def test_the_graph_fixture_contains_exactly_the_expected_callers(temp_repo):
    """
    The expected callers are ground truth for `graph_recall`. If the fixture and the expectation
    drift apart, the score measures the drift rather than the answer.
    """
    from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent

    _, repo_id, _ = temp_repo(ds.build_fixture_files())
    facts = Agent._build_graph_facts(
        "DEPENDENCY_ANALYSIS", ds.GRAPH_FIXTURE_SYMBOL, repo_id, {}, {}
    )

    assert f"({len(ds.GRAPH_EXPECTED_CALLERS)} total)" in facts
    for caller in ds.GRAPH_EXPECTED_CALLERS:
        assert caller in facts, f"{caller} is expected ground truth but is not in the graph"


def test_the_expected_caller_count_is_thirteen():
    """
    The case is only interesting because thirteen exceeds what retrieval returns. If this ever
    shrinks to something retrieval covers, the dataset stops testing the thing it exists for.
    """
    assert len(ds.GRAPH_EXPECTED_CALLERS) == 13
    assert len(set(ds.GRAPH_EXPECTED_CALLERS)) == 13


# ------------------------------------------------------------------ the items are well-formed

def test_item_ids_are_stable_across_calls():
    """Re-seeding must update items, not duplicate them, and that rests entirely on the id."""
    assert [i["id"] for i in ds.intent_items()] == [i["id"] for i in ds.intent_items()]


def test_item_ids_are_unique_within_a_dataset():
    for name, spec in ds.DATASETS.items():
        ids = [item["id"] for item in spec["items"]()]
        assert len(ids) == len(set(ids)), f"duplicate item id in {name}"


def test_the_same_input_in_two_datasets_gets_different_ids():
    """Ids are namespaced by dataset, so an identical question in both does not collide."""
    payload = {"query": "Who calls record_audit?"}
    assert ds.item_id(ds.INTENT_DATASET, payload) != ds.item_id(ds.GRAPH_DATASET, payload)


def test_changing_an_input_changes_its_id():
    a = ds.item_id(ds.INTENT_DATASET, {"query": "Where is auth?"})
    b = ds.item_id(ds.INTENT_DATASET, {"query": "Where is authz?"})
    assert a != b


def test_every_item_says_which_check_applies():
    """
    Two kinds of expectation share the intent dataset. A runner that ignored `check` and compared
    intents would mark all seven trap cases failed — they have no correct intent, only a
    requirement that nothing matched.
    """
    for name, spec in ds.DATASETS.items():
        for item in spec["items"]():
            assert item["metadata"].get("check"), f"{name} has an item with no check"
            assert item["input"].get("query"), f"{name} has an item with no query"
            assert item["expected_output"], f"{name} has an item with no expectation"


def test_trap_items_do_not_claim_an_expected_intent():
    """A trap has no correct intent. Asserting one would be inventing ground truth."""
    traps = [i for i in ds.intent_items() if i["metadata"]["check"] == "no_fragment_match"]
    assert len(traps) == len(ds.INTENT_TRAP_CASES)
    for item in traps:
        assert "intent" not in item["expected_output"]
        assert item["expected_output"]["keyword_score"] == 0


def test_both_datasets_are_registered_and_described():
    assert set(ds.DATASETS) == {ds.INTENT_DATASET, ds.GRAPH_DATASET}
    for name, spec in ds.DATASETS.items():
        assert len(spec["description"]) > 40, f"{name} needs a description worth reading"
        assert spec["items"](), f"{name} is empty"
