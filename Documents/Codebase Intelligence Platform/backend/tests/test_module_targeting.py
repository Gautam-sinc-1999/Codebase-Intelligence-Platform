"""
Questions about a file or module (G-06).

Asking about a *module* is an ordinary way to ask, and there was no path to the right answer for
it: `_extract_target_symbol` matches symbol names only, a module is not a symbol, so the query
matched nothing and the caller fell back to the top-ranked retrieval hit. In the case that
produced this issue that hit was a helper in a test file, and the impact report was about the
test helper rather than the module.
"""
import pytest

from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent


@pytest.fixture
def module_repo(temp_repo):
    """A module, a same-named test file, and two callers outside the module."""
    return temp_repo({
        "src/ui/decide.py": (
            "def choose_route(state):\n    return rank_options(state)\n\n"
            "def rank_options(state):\n    return sorted(state)\n"
        ),
        "src/ui/flow.py": (
            "from src.ui.decide import choose_route\n\n"
            "def run_flow(state):\n    return choose_route(state)\n"
        ),
        "src/ui/handler.py": (
            "from src.ui.decide import rank_options\n\n"
            "def handle(x):\n    return rank_options(x)\n"
        ),
        "tests/test_decide.py": (
            "def a_decision():\n    return {'x': 1}\n\n"
            "def test_choose():\n    assert a_decision()\n"
        ),
    })


# ------------------------------------------------------------------ resolving the file

@pytest.mark.parametrize("query", [
    "If I change the decide module, what will be affected?",
    "What does decide.py do?",
    "Explain src/ui/decide.py",
    "Walk me through the decide module",
])
def test_a_module_question_resolves_to_the_module(module_repo, query):
    _, _, chunks = module_repo
    assert Agent._extract_target_file(query, chunks) == "src/ui/decide.py"


def test_the_implementation_wins_over_the_same_named_test_file(module_repo):
    """
    `test_decide.py` contains the token and used to win by ranking. Its stem is `test_decide`,
    not `decide`, so a stem match now separates them — the tie-break away from tests is only a
    tie-break.
    """
    _, _, chunks = module_repo
    resolved = Agent._extract_target_file("what is in the decide module?", chunks)
    assert resolved == "src/ui/decide.py"
    assert "test" not in resolved


def test_an_explicitly_named_test_file_still_resolves_to_it(module_repo):
    """The bias must not become an override — sometimes the test file is the question."""
    _, _, chunks = module_repo
    assert Agent._extract_target_file("what does tests/test_decide.py cover?", chunks) \
        == "tests/test_decide.py"


def test_a_named_symbol_keeps_precedence_over_a_file(module_repo):
    """"What calls `rank_options`?" is a question about a symbol, wherever it lives."""
    _, _, chunks = module_repo
    assert Agent._extract_target_symbol("What calls rank_options?", chunks) == "rank_options"


def test_a_question_naming_nothing_resolves_to_no_file(module_repo):
    _, _, chunks = module_repo
    assert Agent._extract_target_file("how does any of this work?", chunks) == ""
    assert Agent._extract_target_file("hello", chunks) == ""


# ------------------------------------------------------------------ module-scoped facts

def test_module_facts_list_what_the_file_defines(module_repo):
    _, repo_id, chunks = module_repo
    facts = Agent._build_graph_facts("CHANGE_IMPACT", "", repo_id, {}, {},
                                     target_file="src/ui/decide.py", all_chunks=chunks)

    assert "SYMBOLS DEFINED IN `src/ui/decide.py` (2 total)" in facts
    assert "choose_route" in facts and "rank_options" in facts


def test_module_facts_list_external_dependents(module_repo):
    """What breaks if this module changes — which is what the question was asking."""
    _, repo_id, chunks = module_repo
    facts = Agent._build_graph_facts("CHANGE_IMPACT", "", repo_id, {}, {},
                                     target_file="src/ui/decide.py", all_chunks=chunks)

    assert "EXTERNAL DEPENDENTS" in facts
    assert "run_flow" in facts and "handle" in facts


def test_calls_inside_the_module_are_not_counted_as_dependents(module_repo):
    """
    `choose_route` calls `rank_options` in the same file. That is internal structure, not
    something that breaks when the module changes, and listing it inflates the blast radius.
    """
    _, repo_id, chunks = module_repo
    facts = Agent._build_graph_facts("CHANGE_IMPACT", "", repo_id, {}, {},
                                     target_file="src/ui/decide.py", all_chunks=chunks)

    dependents = facts.split("EXTERNAL DEPENDENTS")[1]
    assert "choose_route` in src/ui/decide.py" not in dependents
    assert "(2 total)" in dependents, "expected exactly the two external callers"


# ------------------------------------------------------------------ end to end

async def test_a_module_question_no_longer_targets_a_test_helper(module_repo):
    """
    The original report in one assertion: the impact report must be about the module asked
    about, not about whatever ranked first.
    """
    _, repo_id, chunks = module_repo

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_module",
        query="If I change the decide module, what will be affected?",
        all_chunks=chunks, _prepare_only=True,
    )

    ctx = prepared["ctx_text"]
    assert "src/ui/decide.py" in ctx
    assert "EXTERNAL DEPENDENTS" in ctx
    assert "a_decision" not in ctx.split("STATIC ANALYSIS FACTS")[1].split("Retrieved Code")[0], \
        "the test helper is still being presented as the subject"


async def test_a_symbol_question_still_produces_symbol_facts(module_repo):
    """The module path must not swallow the case that already worked."""
    _, repo_id, chunks = module_repo

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_symbol",
        query="What calls rank_options?", all_chunks=chunks, _prepare_only=True,
    )

    ctx = prepared["ctx_text"]
    assert "CALLERS OF `rank_options`" in ctx
    assert "SYMBOLS DEFINED IN" not in ctx


# ------------------------------------------------------------------ cases the tie-break cannot save

@pytest.fixture
def sibling_repo(temp_repo):
    """A module and a non-test sibling whose stem *contains* the same token."""
    return temp_repo({
        "src/decide.py": "def choose(x):\n    return x\n",
        "src/decide_helpers.py": "def helper(x):\n    return x\n",
        "src/other.py": "from src.decide import choose\n\ndef run(x):\n    return choose(x)\n",
    })


def test_a_stem_must_match_exactly_not_merely_contain(sibling_repo):
    """
    `decide_helpers` contains `decide` and is not a test file, so the bias away from tests cannot
    break this tie — only an exact stem match can. Without it the winner is whichever file the
    set happened to yield first.
    """
    _, _, chunks = sibling_repo
    assert Agent._extract_target_file("what is in the decide module?", chunks) == "src/decide.py"


def test_the_sibling_is_still_reachable_when_named(sibling_repo):
    _, _, chunks = sibling_repo
    assert Agent._extract_target_file("what is in decide_helpers?", chunks) \
        == "src/decide_helpers.py"


async def test_naming_a_symbol_and_a_file_together_asks_about_the_symbol(sibling_repo):
    """
    "What calls `choose` in decide.py?" names both. It is a question about `choose`; treating it
    as a module question would answer something narrower than what was asked, and the file
    extractor does resolve here — so only the precedence guard prevents it.
    """
    _, repo_id, chunks = sibling_repo

    assert Agent._extract_target_symbol("What calls choose in decide.py?", chunks) == "choose"
    assert Agent._extract_target_file("What calls choose in decide.py?", chunks) == "src/decide.py"

    prepared = await Agent.process_user_query(
        repository_id=repo_id, conversation_id="conv_both",
        query="What calls choose in decide.py?", all_chunks=chunks, _prepare_only=True,
    )

    ctx = prepared["ctx_text"]
    assert "CALLERS OF `choose`" in ctx
    assert "SYMBOLS DEFINED IN" not in ctx, "the file took precedence over the named symbol"


def test_file_resolution_is_deterministic(sibling_repo):
    """
    Candidate paths were iterated out of a set, so two files scoring equally were separated by
    hash order — which varies between processes. The same question must not be answerable
    differently on two runs of the same repository.
    """
    _, _, chunks = sibling_repo
    answers = {Agent._extract_target_file("what is in the decide module?", chunks)
               for _ in range(20)}
    assert len(answers) == 1, f"resolution varied across calls: {answers}"


def test_the_closest_named_file_wins_a_tie(sibling_repo):
    """`decide` means `decide.py`, not the longer name that merely starts with it."""
    _, _, chunks = sibling_repo
    assert Agent._extract_target_file("decide", chunks) == "src/decide.py"
