"""
Query intent classification (G-02).

Two classes of test here, and the second is the one that matters. The first checks that ordinary
questions land on the right intent — the behaviour that already worked. The second checks the
questions that used to land on the *wrong* intent because matching was substring-based, and those
are the reason this file exists.
"""
import pytest

from app.observability.datasets import INTENT_CASES, INTENT_TRAP_CASES
from app.retrieval.intent_classifier import QueryIntentClassifier as IC


# The cases live in `app/observability/datasets.py` rather than here, because the Langfuse
# evaluation dataset is seeded from the same list. Keeping two copies would let a case be fixed in
# one place and left stale in the other — and the stale one would be the dataset, which is exactly
# where a wrong label is hardest to notice.


# ------------------------------------------------------------------ ordinary questions

@pytest.mark.parametrize("query,intent", INTENT_CASES)
def test_ordinary_questions_land_on_the_right_intent(query, intent):
    assert IC.classify(query)["intent"] == intent


# ------------------------------------------------------------------ the substring bug

@pytest.mark.parametrize("query,fragment", INTENT_TRAP_CASES)
def test_a_keyword_buried_inside_another_word_contributes_nothing(query, fragment):
    """
    Each of these routed retrieval down the wrong path before a single chunk was fetched.

    The assertion is on **score**, not on the resulting intent. Asserting "not
    FEATURE_EXPLANATION" would be unsatisfiable here, because FEATURE_EXPLANATION is also the
    fallback — a question that matches nothing correctly lands there. Score is what actually
    distinguishes "matched the wrong keyword" from "matched nothing and fell back", and it is the
    former that was the bug.
    """
    assert IC.classify(query)["score"] == 0, f"still scoring on a fragment: {fragment}"


def test_a_fragment_does_not_outrank_a_genuine_keyword_elsewhere():
    """
    'exchange' contains 'change', but this is a location question and should be read as one —
    a case where the old classifier returned CHANGE_IMPACT outright.
    """
    result = IC.classify("Where does the exchange rate come from?")
    assert result["intent"] == "CODE_LOCATION"


def test_a_filename_is_not_read_as_a_keyword():
    """
    The repository behind this issue contains a module named `impact.py`. Asking about the file
    is a question about code location, not a change-impact analysis.
    """
    assert IC.classify("What does impact.py do?")["intent"] != "CHANGE_IMPACT"
    assert IC.classify("Where is impact.py?")["intent"] == "CODE_LOCATION"
    # The bare word still means what it means.
    assert IC.classify("What is the impact of this?")["intent"] == "CHANGE_IMPACT"


# ------------------------------------------------------------------ scoring, not first-match

def test_the_strongest_signal_wins_not_the_first_branch():
    """
    Previously the cascade returned on the first matching list, so intent depended on the order
    the `if` statements happened to be written in rather than on the question.
    """
    # 'who calls' (2 tokens) outweighs the single word 'change'.
    assert IC.classify("After I change this, who calls it?")["intent"] == "DEPENDENCY_ANALYSIS"
    # And the reverse holds when the impact signal is the stronger one.
    assert IC.classify("What breaks if I change the caller?")["intent"] == "CHANGE_IMPACT"


def test_a_multi_word_phrase_outscores_a_single_word():
    single = IC.classify("Show me the flow")
    phrase = IC.classify("How does it work")
    assert phrase["score"] > single["score"]


def test_an_exact_tie_falls_back_to_declared_precedence():
    """Ties are resolved by declaration order, preserving the original cascade's precedence."""
    result = IC.classify("change depends")
    assert result["intent"] == "CHANGE_IMPACT"


# ------------------------------------------------------------------ degenerate input

@pytest.mark.parametrize("query", ["", "   ", "?!", None])
def test_empty_or_meaningless_input_falls_back_without_raising(query):
    result = IC.classify(query)
    assert result["intent"] == "FEATURE_EXPLANATION"
    assert result["score"] == 0


def test_an_unrecognised_question_falls_back_rather_than_guessing():
    result = IC.classify("Tell me something interesting")
    assert result["intent"] == "FEATURE_EXPLANATION"
    assert result["score"] == 0
    assert result["confidence"] == IC.DEFAULT_CONFIDENCE


def test_case_and_punctuation_are_irrelevant():
    assert IC.classify("WHERE IS the parser?!")["intent"] == "CODE_LOCATION"
    assert IC.classify("where   is  the parser")["intent"] == "CODE_LOCATION"
