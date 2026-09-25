"""
Executing an evaluation dataset.

Langfuse stores, links and compares runs; it does **not** execute the application. The loop is
ours: pull items, call the code, score the output, link each trace to a named run. This module is
that loop, with the Langfuse plumbing deliberately left out — everything here runs, and is tested,
with no service reachable. `scripts/run_dataset.py` is the thin layer that pushes results.

Two task modes, because they answer different questions and cost different amounts:

- **`facts`** (default) — runs intent classification and the graph facts builder. Deterministic,
  free, no LLM. This is what catches the regressions the datasets were built from: G-02's substring
  matching and G-01's missing graph join. Safe to run on every commit.
- **`answer`** — runs the full orchestrator, so the output is a real generated answer. Needed for
  anything judging answer *quality*, and the input to the RAGAS harness. Costs one LLM call per
  item and is rate-limited, so it is opt-in.

A dataset run must be repeatable to be worth anything, which is why the default mode has no model
in it at all.
"""
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from app.observability import datasets as ds

logger = logging.getLogger("observability.runner")

# Namespaced so an evaluation fixture can never be mistaken for a real repository, and so the
# cleanup below has an unambiguous target.
EVAL_REPOSITORY_ID = "eval_thirteen_callers"


class GraphFixture:
    """
    The thirteen-caller repository, indexed once per run.

    Indexing is the expensive part, and every graph item asks about the same repository, so it is
    built on first use and torn down at the end. Built into a temp directory and purged from all
    stores afterwards — an evaluation run must not leave a repository behind that the sidebar then
    offers someone.
    """

    def __init__(self, repository_id: str = EVAL_REPOSITORY_ID):
        self.repository_id = repository_id
        self._root: Optional[str] = None

    def ensure(self) -> str:
        if self._root is not None:
            return self.repository_id

        from app.api.repositories import index_repository_folder

        root = tempfile.mkdtemp(prefix="eval_fixture_")
        for relative, content in ds.build_fixture_files().items():
            target = os.path.join(root, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(content)

        index_repository_folder(root, self.repository_id, "eval-thirteen-callers",
                                trigger="evaluation")
        self._root = root
        return self.repository_id

    def cleanup(self) -> None:
        """Removes the fixture from disk and from every store it was written to."""
        if self._root is None:
            return
        from app.api.repositories import purge_repository_stores

        try:
            purge_repository_stores(self.repository_id)
        except Exception as e:
            logger.warning("Could not purge the evaluation fixture: %s", e)
        shutil.rmtree(self._root, ignore_errors=True)
        self._root = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.cleanup()
        return False


# ------------------------------------------------------------------ executing one item

def execute(item_input: Dict[str, Any], metadata: Dict[str, Any], *,
            mode: str = "facts", fixture: Optional[GraphFixture] = None) -> Dict[str, Any]:
    """
    Runs the code under test for one dataset item and returns its output.

    Dispatches on `metadata["check"]` rather than guessing from the input shape, because the two
    intent checks take identical inputs and differ only in what counts as correct.
    """
    check = metadata.get("check")
    query = item_input.get("query", "")

    if check in ("intent", "no_fragment_match"):
        from app.retrieval.intent_classifier import QueryIntentClassifier

        verdict = QueryIntentClassifier.classify(query)
        return {"intent": verdict.get("intent"), "keyword_score": verdict.get("score")}

    if check == "graph_recall":
        if fixture is None:
            raise ValueError("graph_recall items need a GraphFixture to ask about")
        repository_id = fixture.ensure()
        symbol = item_input.get("symbol", "")

        if mode == "answer":
            raise ValueError(
                "answer mode runs the orchestrator, which is async — use `await aexecute(...)`. "
                "This function stays synchronous because the facts path has no await in it and "
                "is called from sync code."
            )
        return _facts_output(query, symbol, repository_id, metadata)

    raise ValueError(f"No execution path for check '{check}'")


def _facts_output(query: str, symbol: str, repository_id: str,
                  metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The callers the graph found — what the model would be told, before any model is involved."""
    from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent
    from app.graph.neo4j_client import neo4j_client

    callers = neo4j_client.get_callers(symbol, repository_id) or []
    names = sorted({c.get("caller_symbol", "") for c in callers if c.get("caller_symbol")})
    facts = Agent._build_graph_facts(
        metadata.get("intent", "DEPENDENCY_ANALYSIS"), symbol, repository_id, {}, {}
    )
    return {"callers": names, "caller_count": len(names), "facts": facts, "mode": "facts"}


async def aexecute(item_input: Dict[str, Any], metadata: Dict[str, Any], *,
                   mode: str = "facts", fixture: Optional["GraphFixture"] = None) -> Dict[str, Any]:
    """
    The async form of `execute`, required for `answer` mode.

    Generating an answer runs the orchestrator, which is a coroutine. The first version called it
    with `asyncio.run` from inside `execute`, which works from a script and explodes with "cannot
    be called from a running event loop" the moment a caller is already async — which the RAGAS
    harness is. The two forms are kept separate rather than merged because the facts path has no
    await in it and is called from synchronous code.
    """
    check = metadata.get("check")
    if check == "graph_recall" and mode == "answer":
        if fixture is None:
            raise ValueError("graph_recall items need a GraphFixture to ask about")
        return await _answer_output(item_input.get("query", ""),
                                    item_input.get("symbol", ""), fixture.ensure())
    return execute(item_input, metadata, mode=mode, fixture=fixture)


async def _answer_output(query: str, symbol: str, repository_id: str) -> Dict[str, Any]:
    """A real generated answer, for the metrics that can only be computed from prose."""
    from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent
    from app.api.repositories import get_repository_chunks

    chunks = get_repository_chunks(repository_id)
    result = await (Agent.process_user_query(
        repository_id=repository_id,
        conversation_id=f"eval_{abs(hash(query)) % 10_000_000}",
        query=query,
        all_chunks=chunks,
        existing_summary="",
        working_context={},
        history=[],
    ))
    graph_callers = result.get("graph_callers", []) or []

    # The facts block is rebuilt here because `process_user_query` does not return it, and the
    # metrics need the *whole* context the model saw. Without it, every claim the answer takes
    # from the graph looks unsupported: `context_entity_recall` reported 5 of 13 symbols present
    # when all 13 were in the prompt, and `faithfulness` would have marked a correct answer down
    # for the same reason. Scoring against a partial view of the context is not a strict judge,
    # it is a wrong one.
    from app.agents.orchestrator import CodebaseAgentOrchestrator as _Agent
    try:
        facts = _Agent._build_graph_facts("DEPENDENCY_ANALYSIS", symbol, repository_id, {}, {})
    except Exception as e:
        logger.warning("Could not rebuild the facts block for scoring: %s", e)
        facts = ""

    return {
        "answer": result.get("answer", ""),
        "facts": facts,
        "sources": result.get("sources", []),
        "intent": result.get("intent"),
        "degraded": bool(result.get("degraded")),
        "callers": sorted({c.get("caller_symbol", "") for c in graph_callers
                           if c.get("caller_symbol")}),
        "mode": "answer",
    }


# ------------------------------------------------------------------ scoring one item

def evaluate(output: Dict[str, Any], expected: Dict[str, Any],
             metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scores one item's output against its ground truth.

    Returns plain dicts rather than SDK objects so this is usable — and testable — with no Langfuse
    installed. The script converts them at the boundary.
    """
    check = metadata.get("check")

    if check == "intent":
        correct = output.get("intent") == expected.get("intent")
        return [{
            "name": "intent_correct",
            "value": 1.0 if correct else 0.0,
            "comment": (f"expected {expected.get('intent')}, got {output.get('intent')}"
                        if not correct else None),
        }]

    if check == "no_fragment_match":
        # Zero means nothing matched. A non-zero score is the bug, whatever intent came out.
        score = output.get("keyword_score")
        clean = score == 0
        return [{
            "name": "no_fragment_match",
            "value": 1.0 if clean else 0.0,
            "comment": (f"scored {score} on a fragment: {metadata.get('reason', '')}"
                        if not clean else None),
        }]

    if check == "graph_recall":
        return _graph_evaluations(output, expected)

    raise ValueError(f"No evaluator for check '{check}'")


def _graph_evaluations(output: Dict[str, Any], expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    How much of the known-complete caller set the system found, and invented.

    Deliberately named `graph_caller_*` rather than `graph_recall`. The trace score of that name
    measures whether the *answer text* mentioned the callers; this measures whether the *graph*
    found them. Sharing a name would silently average two different quantities on one dashboard.
    """
    truth = set(expected.get("callers", []))
    found = set(output.get("callers", []))
    if not truth:
        return []

    hits = truth & found
    evaluations = [
        {
            "name": "graph_caller_recall",
            "value": len(hits) / len(truth),
            "comment": (f"missing: {sorted(truth - found)}" if truth - found else None),
        },
        {
            "name": "graph_caller_precision",
            "value": len(hits) / len(found) if found else 0.0,
            "comment": (f"invented: {sorted(found - truth)}" if found - truth else None),
        },
    ]

    # In `answer` mode there is prose to check as well: the graph can hold all thirteen while the
    # answer names five, which is exactly the failure G-01 was about.
    if output.get("mode") == "answer":
        from app.observability import scorers

        answer = output.get("answer", "")
        named = [c for c in truth if scorers._mentions(answer, c)]
        evaluations.append({
            "name": "answer_names_callers",
            "value": len(named) / len(truth),
            "comment": f"{len(named)} of {len(truth)} callers appear in the answer",
        })
        evaluations.append({
            "name": "degraded",
            "value": 0.0 if output.get("degraded") else 1.0,
            "comment": "answer came from the template, not a model" if output.get("degraded") else None,
        })

    return evaluations


# ------------------------------------------------------------------ the whole run

def run_items(items: List[Dict[str, Any]], *, mode: str = "facts",
              fixture: Optional[GraphFixture] = None) -> List[Dict[str, Any]]:
    """
    Executes and scores every item, collecting failures rather than raising on them.

    One item blowing up must not abandon the rest — a run that stops at item 3 of 23 reports
    nothing useful about the other 20, and the usual cause is a fixture problem rather than a
    regression.
    """
    results = []
    for item in items:
        record: Dict[str, Any] = {
            "id": item.get("id"),
            "query": item.get("input", {}).get("query", ""),
            "check": item.get("metadata", {}).get("check"),
        }
        try:
            output = execute(item["input"], item.get("metadata", {}),
                             mode=mode, fixture=fixture)
            evaluations = evaluate(output, item.get("expected_output", {}),
                                   item.get("metadata", {}))
            record["output"] = output
            record["evaluations"] = evaluations
            record["passed"] = all(e["value"] == 1.0 for e in evaluations) if evaluations else None
        except Exception as e:
            logger.warning("Item %s failed to run: %s", record["id"], e)
            record["error"] = str(e)
            record["evaluations"] = []
            record["passed"] = False
        results.append(record)
    return results


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates a run: per-metric means, and the items that failed."""
    totals: Dict[str, List[float]] = {}
    for record in results:
        for evaluation in record.get("evaluations", []):
            totals.setdefault(evaluation["name"], []).append(float(evaluation["value"]))

    return {
        "items": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if r.get("passed") is False),
        "errors": sum(1 for r in results if r.get("error")),
        "metrics": {name: sum(vals) / len(vals) for name, vals in sorted(totals.items()) if vals},
        "failures": [
            {"query": r["query"], "check": r["check"],
             "why": r.get("error") or _first_comment(r)}
            for r in results if r.get("passed") is False
        ],
    }


def _first_comment(record: Dict[str, Any]) -> str:
    for evaluation in record.get("evaluations", []):
        if evaluation["value"] != 1.0 and evaluation.get("comment"):
            return evaluation["comment"]
    return ""
