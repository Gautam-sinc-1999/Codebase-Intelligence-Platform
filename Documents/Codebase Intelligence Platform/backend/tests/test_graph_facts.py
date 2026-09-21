"""
Static analysis facts in the prompt (G-01).

The call graph was built, queried, and rendered in the UI — and then the answer was generated
from retrieved snippets alone. These tests cover the join that was missing, and the property that
makes it worth having: retrieval returns the top `k` chunks, so for a widely-used symbol the
snippets show a handful of call sites while the graph has all of them.
"""
import pytest

from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent


@pytest.fixture
def wide_repo(temp_repo):
    """One helper called from many files — more call sites than retrieval will ever return."""
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
    return temp_repo(files)


# ------------------------------------------------------------------ the facts block

def test_callers_reach_the_prompt(wide_repo):
    _, repo_id, _ = wide_repo
    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})

    assert "CALLERS OF `record_audit`" in facts
    assert "handle_request" in facts
    for i in range(1, 13):
        assert f"operation_{i}" in facts, f"operation_{i} missing from the facts block"


def test_the_facts_carry_a_total_and_claim_completeness(wide_repo):
    """
    The count is the point. Without it the model reports whatever it can see and calls that the
    answer; with it, an answer listing five of thirteen is self-evidently wrong.
    """
    _, repo_id, _ = wide_repo
    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})

    assert "(13 total)" in facts
    assert "this list is complete" in facts
    assert "authoritative" in facts.lower()


def test_the_facts_say_the_snippets_are_an_excerpt(wide_repo):
    """The model must be told *why* the two sources differ, or it will average them."""
    _, repo_id, _ = wide_repo
    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})
    assert "excerpt" in facts
    assert "the facts above are correct" in facts


def test_facts_beat_retrieval_on_exactly_the_case_that_matters(wide_repo):
    """
    Quantifies the gap this fix closes: the graph knows every call site, retrieval returns `k`.
    """
    from app.retrieval.hybrid_retriever import HybridRetriever

    _, repo_id, chunks = wide_repo
    retrieved = HybridRetriever.retrieve(
        repository_id=repo_id, query="What calls record_audit?", all_chunks=chunks, top_k=6)
    retrieved_symbols = {c["symbol"] for c in retrieved}

    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})
    in_facts = sum(1 for i in range(1, 13) if f"operation_{i}" in facts)
    in_snippets = sum(1 for i in range(1, 13) if f"operation_{i}" in retrieved_symbols)

    assert in_facts == 12
    assert in_snippets < in_facts, "the fixture no longer exercises the gap it exists to show"


# ------------------------------------------------------------------ placement and shape

def test_facts_appear_before_the_snippets_in_the_context(wide_repo):
    """
    Placement is deliberate: the snippet block can be long, and anything after it competes for
    attention with thousands of characters of source.
    """
    _, repo_id, _ = wide_repo
    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})
    sources = [{"file_path": "a.py", "symbol": "x", "start_line": 1, "end_line": 2,
                "entity_type": "function", "code_snippet": "def x():\n    pass"}]

    ctx = Agent._build_context("q", "", "", sources, facts)
    assert ctx.index("STATIC ANALYSIS FACTS") < ctx.index("Retrieved Code Context")


def test_facts_survive_retrieval_finding_nothing(wide_repo):
    """
    The graph can answer when retrieval cannot, and previously that case discarded everything and
    told the user nothing was found.
    """
    _, repo_id, _ = wide_repo
    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})

    ctx = Agent._build_context("q", "", "", [], facts)
    assert "record_audit" in ctx
    assert "still stand" in ctx
    assert "ask them to name a file" not in ctx


def test_nothing_is_emitted_when_the_graph_has_nothing_to_say():
    """An empty heading is worse than no heading: it invites the model to fill it in."""
    assert Agent._build_graph_facts("FEATURE_EXPLANATION", "", "", {}, {}) == ""
    assert Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "no_such_symbol", "no_such_repo", {}, {}) == ""


def test_a_huge_caller_list_is_truncated_and_says_so(monkeypatch, wide_repo):
    """A helper with hundreds of callers must not consume the budget the source code needs."""
    _, repo_id, _ = wide_repo
    monkeypatch.setattr(Agent, "MAX_GRAPH_FACTS", 3)

    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})
    assert "(13 total)" in facts, "truncation must not hide the real count"
    assert "and 10 more" in facts
    assert "the total above is exact" in facts


# ------------------------------------------------------------------ the other two analyses

def test_change_impact_reaches_the_prompt(wide_repo):
    _, repo_id, chunks = wide_repo
    from app.agents.impact_analyzer import ChangeImpactAnalyzer

    impact = ChangeImpactAnalyzer.analyze_change_impact("record_audit", chunks)
    facts = Agent._build_graph_facts("CHANGE_IMPACT", "record_audit", repo_id, {}, impact)

    assert "CHANGE TARGET" in facts
    assert "CONFIRMED CALLERS" in facts


def test_execution_flow_reaches_the_prompt():
    flow = {
        "feature": "checkout",
        "flow_steps": [
            {"step": 1, "layer": "Frontend Component", "title": "Component: Checkout",
             "file_path": "ui/Checkout.jsx", "lines": "4-51"},
            {"step": 2, "layer": "Business Service", "title": "Service: DiscountService",
             "file_path": "svc/discount.py", "lines": "10-40"},
        ],
    }
    facts = Agent._build_graph_facts("FEATURE_EXPLANATION", "", "", flow, {})
    assert "TRACED EXECUTION FLOW" in facts
    assert "ui/Checkout.jsx:4-51" in facts
    assert "step 2 — Business Service" in facts


def test_a_graph_outage_does_not_break_answering(wide_repo, monkeypatch):
    """
    Graph facts improve an answer; failing to fetch them must not prevent one.

    Asserted per-lookup rather than on the whole block. The two lookups are independent, and on
    the in-memory backend the surviving one still returns data — which is correct behaviour, so
    demanding an empty block would be asserting that one failure suppresses everything.
    """
    from app.graph import neo4j_client as graph_module

    _, repo_id, _ = wide_repo

    def explode(*args, **kwargs):
        raise RuntimeError("graph down")

    monkeypatch.setattr(graph_module.neo4j_client, "get_callers", explode)

    facts = Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {})
    assert "CALLERS OF" not in facts, "a failed lookup contributed content anyway"

    monkeypatch.setattr(graph_module.neo4j_client, "get_forward_dependencies", explode)
    assert Agent._build_graph_facts("DEPENDENCY_ANALYSIS", "record_audit", repo_id, {}, {}) == ""


# ------------------------------------------------------------------ the prompt itself

async def test_the_assembled_prompt_states_which_source_wins(wide_repo):
    """
    Without an explicit precedence rule the model averages two sources that disagree — and the
    snippets, being longer and more vivid, tend to win.

    Built through the real preparation path rather than by reading the source, so the assertion
    covers what is actually sent.
    """
    _, repo_id, chunks = wide_repo

    prepared = await Agent.process_user_query(
        repository_id=repo_id,
        conversation_id="conv_prompt_probe",
        query="What calls record_audit?",
        all_chunks=chunks,
        _prepare_only=True,
    )

    assert "AUTHORITATIVE and COMPLETE" in prepared["sys_prompt"]
    assert "follow the facts" in prepared["sys_prompt"]
    assert "not the whole repository" in prepared["sys_prompt"]


async def test_a_dependency_question_now_carries_graph_data(wide_repo):
    """
    The narrowest statement of the bug: only CHANGE_IMPACT and FEATURE_EXPLANATION computed
    anything, so "who calls X?" — the question a call graph exists to answer — reached the model
    with no graph data at all.
    """
    _, repo_id, chunks = wide_repo

    prepared = await Agent.process_user_query(
        repository_id=repo_id,
        conversation_id="conv_dep_probe",
        query="What calls record_audit?",
        all_chunks=chunks,
        _prepare_only=True,
    )

    ctx = prepared["ctx_text"]
    assert "STATIC ANALYSIS FACTS" in ctx
    assert "(13 total)" in ctx
    named = sum(1 for i in range(1, 13) if f"operation_{i}" in ctx)
    assert named == 12, f"only {named} of 12 callers reached the prompt"


# ------------------------------------------------------------------ not volunteering facts

async def test_a_greeting_does_not_collect_an_authoritative_caller_list(wide_repo):
    """
    `current_symbol` falls back to the top retrieval hit, so before this gate even "hello" picked
    up a complete caller list for whatever happened to rank first — a confident answer to a
    question nobody asked, and a good way to derail a greeting.
    """
    _, repo_id, chunks = wide_repo

    for greeting in ("hello", "thanks!"):
        prepared = await Agent.process_user_query(
            repository_id=repo_id, conversation_id="conv_greeting",
            query=greeting, all_chunks=chunks, _prepare_only=True,
        )
        assert "STATIC ANALYSIS FACTS" not in prepared["ctx_text"], \
            f"{greeting!r} collected graph facts about an unmentioned symbol"


async def test_an_explicitly_named_symbol_still_gets_facts(wide_repo):
    _, repo_id, chunks = wide_repo
    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_named",
        query="Explain record_audit", all_chunks=chunks, _prepare_only=True,
    )
    assert "STATIC ANALYSIS FACTS" in prepared["ctx_text"]


async def test_a_dependency_question_gets_facts_even_from_an_inferred_subject(wide_repo):
    """
    For a dependency or impact question the subject *is* the question, so the best available
    guess is worth using — unlike a greeting, where there is no question to be the subject of.
    """
    _, repo_id, chunks = wide_repo
    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_inferred",
        query="what does this depend on?", all_chunks=chunks, _prepare_only=True,
    )
    assert prepared["intent"] == "DEPENDENCY_ANALYSIS"
    assert "STATIC ANALYSIS FACTS" in prepared["ctx_text"]
