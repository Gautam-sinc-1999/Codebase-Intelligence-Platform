"""
Prompt budgeting and honest degradation (G-13).

Two separate failures behind one symptom. A prompt that consumes the whole per-minute token
budget does not fail outright — it succeeds once and rate-limits everything for the next minute,
which is how one query in four fell back to a template. And when it did fall back, the payload
was indistinguishable from a generated answer, so the user saw a visibly poorer response with
nothing indicating why.
"""
import pytest

from app.core.config import settings
from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent


@pytest.fixture
def sprawling_repo(temp_repo):
    """One helper called from many files with long names — an ordinary shared utility."""
    files = {"services/internal/common/audit.py":
             "def record_audit(event, actor):\n    return event\n"}
    for i in range(60):
        files[f"services/internal/common/business_operation_module_{i}.py"] = (
            "from services.internal.common.audit import record_audit\n\n"
            f"def perform_business_operation_number_{i}(actor, payload):\n"
            f"    record_audit('op_{i}', actor)\n    return payload\n"
        )
    return temp_repo(files)


def _long_history():
    return [{"role": "user", "content": "x" * 600},
            {"role": "assistant", "content": "y" * 600}] * 3


# ------------------------------------------------------------------ the budget holds

async def test_the_whole_prompt_stays_within_the_token_budget(sprawling_repo):
    _, repo_id, chunks = sprawling_repo
    history = _long_history()

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_budget",
        query="If I change record_audit what breaks?",
        all_chunks=chunks, history=history, _prepare_only=True,
    )

    messages = Agent._build_messages(prepared["sys_prompt"], prepared["ctx_text"], history)
    total_chars = sum(len(m["content"]) for m in messages)
    estimated_tokens = total_chars // Agent.CHARS_PER_TOKEN

    assert estimated_tokens <= settings.MAX_PROMPT_TOKENS, (
        f"prompt is ~{estimated_tokens} tokens against a {settings.MAX_PROMPT_TOKENS} budget")


async def test_the_budget_holds_for_every_intent(sprawling_repo):
    _, repo_id, chunks = sprawling_repo
    history = _long_history()

    for query in ["How does audit work?",
                  "What calls record_audit?",
                  "If I change record_audit what breaks?",
                  "Where is record_audit?"]:
        prepared = await Agent.process_user_query(
            repository_id=repo_id, conversation_id="conv_all",
            query=query, all_chunks=chunks, history=history, _prepare_only=True)
        messages = Agent._build_messages(prepared["sys_prompt"], prepared["ctx_text"], history)
        total = sum(len(m["content"]) for m in messages)
        assert total // Agent.CHARS_PER_TOKEN <= settings.MAX_PROMPT_TOKENS, query


async def test_the_facts_block_is_capped_in_aggregate_not_only_per_block(sprawling_repo):
    """
    `MAX_GRAPH_FACTS` caps each block; there can be six. On a helper called from 120 files the
    three qualifying blocks produced 11,684 characters between them — about 2,900 tokens before
    any source code was added.
    """
    _, repo_id, chunks = sprawling_repo

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_facts",
        query="If I change record_audit what breaks?", all_chunks=chunks, _prepare_only=True)

    facts = prepared["ctx_text"].split("Retrieved Code Context")[0]
    cap = int(Agent._char_budget() * Agent.FACTS_BUDGET_SHARE)
    assert len(facts) <= cap * 1.1, f"facts block is {len(facts)} against a {cap} share"


async def test_truncation_is_stated_and_the_totals_stay_exact(sprawling_repo):
    """
    Silent truncation would be worse than the overrun: a list that stops mid-way reads as a
    complete one, which is the precise failure the facts block exists to prevent.
    """
    _, repo_id, chunks = sprawling_repo

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_trunc",
        query="If I change record_audit what breaks?", all_chunks=chunks, _prepare_only=True)

    facts = prepared["ctx_text"].split("Retrieved Code Context")[0]
    assert "truncated to fit" in facts
    assert "(60 total)" in facts or "(61 total)" in facts, "the exact count was lost"


async def test_code_still_reaches_the_prompt_after_the_facts_take_their_share(sprawling_repo):
    """
    Code is what stops the model inventing behaviour, so it takes the remainder rather than a
    fixed slice — but the remainder must never be nothing.
    """
    _, repo_id, chunks = sprawling_repo

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_code",
        query="If I change record_audit what breaks?",
        all_chunks=chunks, history=_long_history(), _prepare_only=True)

    ctx = prepared["ctx_text"]
    assert "Retrieved Code Context" in ctx
    code = ctx.split("Retrieved Code Context")[1]
    assert "def " in code, "no source survived the budget"
    assert len(code) > 1000


async def test_a_small_repository_is_not_trimmed_at_all(temp_repo):
    """The budget must not degrade the common case to protect the rare one."""
    _, repo_id, chunks = temp_repo({"a.py": "def alpha():\n    return 1\n"})

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_small",
        query="What calls alpha?", all_chunks=chunks, _prepare_only=True)

    assert "truncated to fit" not in prepared["ctx_text"]


def test_the_budget_is_configurable_and_has_a_floor(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PROMPT_TOKENS", 10000)
    assert Agent._char_budget() == 40000

    monkeypatch.setattr(settings, "MAX_PROMPT_TOKENS", 1)
    assert Agent._char_budget() >= 2000, "a nonsensical setting must not empty the prompt"


# ------------------------------------------------------------------ honest degradation

async def test_a_template_answer_is_marked_degraded(api_client, temp_repo):
    """
    The template is genuinely useful — it cites real files and line ranges. What it is not is
    equivalent, and returning it indistinguishably left the user with a worse answer and no
    explanation. Tests run with no provider configured, so this is the path they take.
    """
    _, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})

    conv = api_client.post("/api/conversations",
                           json={"repository_id": repo_id, "title": "degraded"}).json()
    response = api_client.post(f"/api/conversations/{conv['conversation_id']}/messages",
                               json={"message": "What calls alpha?"})

    body = response.json()
    assert body["degraded"] is True
    assert body["degraded_reason"]
    assert "rate-limited" in body["degraded_reason"]
    assert body["answer"], "a degraded answer is still an answer"
    assert body["sources"], "the citations are still real"


async def test_the_marker_survives_into_the_stored_thread(api_client, temp_repo):
    """A reader coming back later should still see which answers were assembled, not written."""
    _, repo_id, _ = temp_repo({"a.py": "def alpha():\n    return 1\n"})

    conv = api_client.post("/api/conversations",
                           json={"repository_id": repo_id, "title": "stored"}).json()
    cid = conv["conversation_id"]
    api_client.post(f"/api/conversations/{cid}/messages", json={"message": "What calls alpha?"})

    stored = api_client.get(f"/api/conversations/{cid}").json()
    assistant = [m for m in stored["messages"] if m["role"] == "assistant"][-1]
    assert assistant["degraded"] is True


async def test_a_generated_answer_is_not_marked_degraded(temp_repo):
    """The flag must be honest in both directions, or a client learns to ignore it."""
    _, repo_id, chunks = temp_repo({"a.py": "def alpha():\n    return 1\n"})

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_ok",
        query="What calls alpha?", all_chunks=chunks, _prepare_only=True)

    result = await Agent._finish(prepared, "A real model wrote this answer.")
    assert result["degraded"] is False
    assert result["degraded_reason"] is None
    assert result["answer"] == "A real model wrote this answer."


@pytest.fixture
def bulky_repo(temp_repo):
    """
    Large function bodies *and* many callers — the combination the budget exists for.

    Retrieval returns the top chunks whole, so a repository of substantial functions can fill the
    code section entirely on its own. With the facts block also competing, a fixed code cap and a
    computed remainder stop agreeing, which is what makes this fixture worth having.
    """
    body = "\n".join(f"    step_{i} = compute_value({i})" for i in range(90))
    files = {"core/engine.py": f"def run_engine(config):\n{body}\n    return step_0\n"}
    for i in range(12):
        files[f"handlers/handler_{i}.py"] = (
            "from core.engine import run_engine\n\n"
            f"def handle_request_{i}(request):\n{body}\n    return run_engine(request)\n"
        )
    return temp_repo(files)


async def test_the_code_section_takes_the_remainder_not_a_fixed_cap(bulky_repo):
    """
    The code section was a flat 12,000 characters regardless of what else was in the prompt, so
    adding the facts block (G-01) raised every prompt with nothing to compensate.

    At the default budget the difference is real but small — about 340 tokens — so a flat cap
    happens to stay under it too. The property worth asserting is therefore the allocation
    itself: code is bounded by what is left, whatever that is.
    """
    _, repo_id, chunks = bulky_repo
    history = _long_history()

    available = sum(len((c.get("code_snippet") or "")) for c in chunks[:6])
    assert available > Agent.MAX_CODE_CHARS_TOTAL, (
        f"fixture only offers {available} characters of code; it cannot distinguish a flat cap "
        f"from a computed remainder")

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_bulky",
        query="If I change run_engine what breaks?",
        all_chunks=chunks, history=history, _prepare_only=True)

    code = prepared["ctx_text"].split("Retrieved Code Context")[1]
    assert len(code) < Agent.MAX_CODE_CHARS_TOTAL, (
        "the code section used its old flat cap rather than the remaining budget")


async def test_the_budget_is_respected_at_any_configured_value(bulky_repo, monkeypatch):
    """
    A budget that only holds at its default is not a budget.

    Set low enough, the old flat code cap alone would consume the entire allowance before the
    system prompt, the facts or the history were counted — which is what makes this the test that
    distinguishes the two implementations rather than one that has slack to absorb the difference.
    """
    _, repo_id, chunks = bulky_repo
    history = _long_history()

    monkeypatch.setattr(settings, "MAX_PROMPT_TOKENS", 3000)
    assert Agent.MAX_CODE_CHARS_TOTAL >= Agent._char_budget(), (
        "precondition: the old flat cap must exceed the whole budget for this to bite")

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_tight",
        query="If I change run_engine what breaks?",
        all_chunks=chunks, history=history, _prepare_only=True)

    messages = Agent._build_messages(prepared["sys_prompt"], prepared["ctx_text"], history)
    total = sum(len(m["content"]) for m in messages)
    assert total // Agent.CHARS_PER_TOKEN <= settings.MAX_PROMPT_TOKENS, (
        f"prompt is ~{total // Agent.CHARS_PER_TOKEN} tokens against a 3000-token budget")


def test_the_oldest_turns_are_dropped_first_not_the_newest():
    """
    When the history budget runs out it must be the oldest exchange that goes. Dropping the most
    recent would discard exactly the turn a follow-up refers back to — "who calls it?" resolves
    against the previous answer, not one from six turns ago.
    """
    history = [{"role": "user", "content": f"TURN{i} " + "x" * 590} for i in range(6)]
    messages = Agent._build_messages("S", "Q", history)

    kept = [m["content"] for m in messages if m["content"].startswith("TURN")]
    assert kept, "the whole history was dropped"
    assert "TURN5" in kept[-1], "the most recent turn was dropped"
    assert kept == sorted(kept, key=lambda c: int(c.split()[0][4:])), "history is out of order"
