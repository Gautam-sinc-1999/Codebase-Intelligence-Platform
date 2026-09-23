"""
Evaluation datasets, defined once and used by both the test suite and the Langfuse seeder.

Most projects start observability with an empty dataset and no labels, then never fill it. This one
starts populated, because the labels already existed as test fixtures — the intent cases from the
G-02 substring-matching work, and the thirteen-caller case from G-01.

**These are the same objects the tests assert against.** `tests/test_intent.py` imports its
parameters from here rather than keeping its own copy, so a case cannot be fixed in one place and
left stale in the other. That is the whole reason this module exists instead of the seeder simply
hardcoding a list.

Items carry a stable `id` derived from their input, so re-seeding updates rather than duplicates.
"""
import hashlib
import json
from typing import Any, Dict, List, Tuple

INTENT_DATASET = "intent-classification"
GRAPH_DATASET = "graph-grounding"


# ------------------------------------------------------------------ intent classification

# Ordinary questions, and the intent each must land on. These are the behaviours that already
# worked; they are here to catch a fix for something else breaking them.
INTENT_CASES: List[Tuple[str, str]] = [
    ("Where is authentication implemented?", "CODE_LOCATION"),
    ("Where are the database migrations?", "CODE_LOCATION"),
    ("Which file holds the retry logic?", "CODE_LOCATION"),
    ("What depends on calculate_discount?", "DEPENDENCY_ANALYSIS"),
    ("Who calls process_checkout?", "DEPENDENCY_ANALYSIS"),
    ("What are the dependencies of the billing module?", "DEPENDENCY_ANALYSIS"),
    ("If I change the discount logic what breaks?", "CHANGE_IMPACT"),
    ("What will be affected if I modify the parser?", "CHANGE_IMPACT"),
    ("How does checkout work?", "FEATURE_EXPLANATION"),
    ("How do sessions get created?", "FEATURE_EXPLANATION"),
    ("Walk me through the payment flow", "FEATURE_EXPLANATION"),
    ("Why does the upload fail?", "DEBUGGING"),
    ("What is the overall architecture?", "ARCHITECTURE"),
    ("Give me an overview of the components", "ARCHITECTURE"),
]

# The cases that motivated the rewrite: a keyword buried inside a longer word used to score, so
# each of these routed retrieval down the wrong path before a single chunk was fetched.
#
# The expectation is on **score**, not on the resulting intent. "Not FEATURE_EXPLANATION" is
# unsatisfiable, because that is also the fallback — a question matching nothing correctly lands
# there. Score is what separates "matched the wrong keyword" from "matched nothing and fell back",
# and only the former was ever the bug.
INTENT_TRAP_CASES: List[Tuple[str, str]] = [
    ("Which parts are unchanged since last release?", "'unchanged' contains 'change'"),
    ("Is the parser independent of the lexer?", "'independent' contains 'depend'"),
    ("Describe the workflow engine", "'workflow' contains 'flow'"),
    ("Summarise the changelog", "'changelog' starts with 'change'"),
    ("What is a changepoint in this codebase?", "'changepoint' starts with 'change'"),
    ("Explain the importer module", "'importer' contains 'import'"),
    ("What does the classifier do?", "no keyword at all, only fragments"),
]


# ------------------------------------------------------------------ graph grounding

# One helper called from thirteen places — more call sites than retrieval will ever return. The
# point of the case: the snippets show a handful of callers while the graph has all of them, so an
# answer naming five of thirteen is self-evidently wrong and `graph_recall` catches it.
GRAPH_FIXTURE_SYMBOL = "record_audit"
GRAPH_EXPECTED_CALLERS = [f"operation_{i}" for i in range(1, 13)] + ["handle_request"]

GRAPH_CASES: List[Dict[str, Any]] = [
    {
        "question": f"What depends on {GRAPH_FIXTURE_SYMBOL}?",
        "intent": "DEPENDENCY_ANALYSIS",
        "symbol": GRAPH_FIXTURE_SYMBOL,
        "expected_callers": GRAPH_EXPECTED_CALLERS,
    },
    {
        "question": f"Who calls {GRAPH_FIXTURE_SYMBOL}?",
        "intent": "DEPENDENCY_ANALYSIS",
        "symbol": GRAPH_FIXTURE_SYMBOL,
        "expected_callers": GRAPH_EXPECTED_CALLERS,
    },
]


def build_fixture_files() -> Dict[str, str]:
    """
    The repository the graph cases are asked about.

    Generated rather than stored so the thirteen callers cannot drift out of step with
    `GRAPH_EXPECTED_CALLERS` — both come from the same range.
    """
    files = {"services/audit.py": "def record_audit(event, actor):\n    return {'e': event}\n"}
    for i in range(1, 13):
        files[f"services/module_{i}.py"] = (
            "from services.audit import record_audit\n\n"
            f"def operation_{i}(actor, payload):\n"
            f"    record_audit('operation_{i}', actor)\n"
            "    return payload\n"
        )
    files["api/handlers.py"] = (
        "from services.audit import record_audit\n\n"
        "def handle_request(request):\n"
        "    record_audit('http_request', request.user)\n"
        "    return {'ok': True}\n"
    )
    return files


# ------------------------------------------------------------------ item construction

def item_id(dataset_name: str, payload: Any) -> str:
    """
    A stable id for an item, derived from its content.

    Idempotence depends on this: Langfuse upserts an item when given an id it already holds, so a
    second seeding run updates the existing items instead of doubling the dataset. Re-running a
    seeder must never change what a dataset measures.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(f"{dataset_name}:{encoded}".encode("utf-8")).hexdigest()
    return digest[:24]


def intent_items() -> List[Dict[str, Any]]:
    """
    The intent dataset as Langfuse items.

    Two kinds of expectation live in one dataset, so each item says in its metadata which check
    applies. A runner that ignored `check` and compared intents would mark every trap case failed,
    because a trap has no correct intent — only a requirement that nothing matched.
    """
    items: List[Dict[str, Any]] = []

    for query, intent in INTENT_CASES:
        payload = {"query": query}
        items.append({
            "id": item_id(INTENT_DATASET, payload),
            "input": payload,
            "expected_output": {"intent": intent},
            "metadata": {"check": "intent", "origin": "G-02"},
        })

    for query, reason in INTENT_TRAP_CASES:
        payload = {"query": query}
        items.append({
            "id": item_id(INTENT_DATASET, payload),
            "input": payload,
            # Zero means no keyword matched. A non-zero score here is the bug, whatever intent
            # the classifier ultimately returned.
            "expected_output": {"keyword_score": 0},
            "metadata": {"check": "no_fragment_match", "reason": reason, "origin": "G-02"},
        })

    return items


def graph_items() -> List[Dict[str, Any]]:
    """The thirteen-caller cases as Langfuse items, with the full caller list as ground truth."""
    items = []
    for case in GRAPH_CASES:
        payload = {"query": case["question"], "symbol": case["symbol"]}
        items.append({
            "id": item_id(GRAPH_DATASET, payload),
            "input": payload,
            "expected_output": {
                "callers": sorted(case["expected_callers"]),
                "caller_count": len(case["expected_callers"]),
            },
            "metadata": {
                "check": "graph_recall",
                "intent": case["intent"],
                "origin": "G-01",
                # The runner builds this repository before asking, so the question has something
                # true to be right about.
                "fixture": "thirteen_callers",
            },
        })
    return items


DATASETS: Dict[str, Dict[str, Any]] = {
    INTENT_DATASET: {
        "description": (
            "Query intent classification. Ordinary questions that must land on the right intent, "
            "plus the substring traps from G-02 where a keyword buried inside a longer word must "
            "score nothing."
        ),
        "items": intent_items,
    },
    GRAPH_DATASET: {
        "description": (
            "Graph grounding. One helper with thirteen callers — more than retrieval returns — so "
            "an answer naming only the retrieved few is measurably incomplete."
        ),
        "items": graph_items,
    },
}
