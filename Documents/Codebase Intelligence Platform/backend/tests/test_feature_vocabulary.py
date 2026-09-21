"""
What counts as a feature name in a repository (G-14).

After discussing `scan_change_point`, the tracked feature became **"point"** — a fragment of one
symbol's name rather than anything the repository is organised around. Cosmetic where the feature
is only a label, but it also seeds `FeatureTracer`, so a bad term produces a weak or empty
execution flow.
"""
import pytest

from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent


@pytest.fixture(autouse=True)
def _clear_vocab_cache():
    Agent._feature_vocab_cache.clear()
    yield
    Agent._feature_vocab_cache.clear()


@pytest.fixture
def scanner_repo(temp_repo):
    """The reported case: one symbol whose name contains a generic fragment."""
    return temp_repo({
        "scanner.py": "def scan_change_point(series):\n    return series\n",
        "other.py": "def unrelated():\n    return 1\n",
    })


@pytest.fixture
def themed_repo(temp_repo):
    """A cross-cutting theme, and a feature that lives in exactly one file."""
    return temp_repo({
        "frontend/Checkout.jsx": "export function CheckoutPanel() { return 1; }\n",
        "api/checkout_controller.py": "def process_checkout(order):\n    return order\n",
        "services/checkout_rules.py": "def apply_rules(order):\n    return order\n",
        "services/payment_service.py": "def charge(amount):\n    return amount\n",
        "models/order.py": "def create(row):\n    return row\n",
    })


# ------------------------------------------------------------------ the reported bug

def test_a_fragment_of_one_symbol_is_not_a_feature(scanner_repo):
    """"point" comes from `scan_change_point` and names nothing this repository is about."""
    _, repo_id, chunks = scanner_repo
    assert Agent._extract_target_feature("what is the point of this?", repo_id, chunks) == ""


def test_the_fragment_is_absent_from_the_vocabulary_itself(scanner_repo):
    _, repo_id, chunks = scanner_repo
    vocabulary = Agent._build_feature_vocabulary(repo_id, chunks)
    assert "point" not in vocabulary
    assert "change" not in vocabulary


def test_naming_the_whole_symbol_still_resolves(scanner_repo):
    """
    Splitting the query alone turned `scan_change_point` into scan/change/point, so once
    fragments stopped being admitted a user naming the symbol outright matched nothing.
    """
    _, repo_id, chunks = scanner_repo
    assert Agent._extract_target_feature("explain scan_change_point", repo_id, chunks) \
        == "scan_change_point"


def test_a_file_stem_is_a_feature_name(scanner_repo):
    _, repo_id, chunks = scanner_repo
    assert Agent._extract_target_feature("how does the scanner work?", repo_id, chunks) == "scanner"


# ------------------------------------------------------------------ real features survive

def test_a_theme_spanning_several_files_is_a_feature(themed_repo):
    """"checkout" is a fragment everywhere it appears, and is unmistakably the feature."""
    _, repo_id, chunks = themed_repo
    assert Agent._extract_target_feature("How does checkout work?", repo_id, chunks) == "checkout"


def test_a_feature_living_in_one_file_still_counts_when_the_file_is_named_for_it(themed_repo):
    """
    `payment` appears in a single file, so a "must span two files" rule alone would discard it.
    It is the leading token of `payment_service.py` — the file is named after the feature.
    """
    _, repo_id, chunks = themed_repo
    assert Agent._extract_target_feature("explain the payment flow", repo_id, chunks) == "payment"


def test_the_longest_match_wins(themed_repo):
    """"checkout" is more specific than "check"."""
    _, repo_id, chunks = themed_repo
    assert Agent._extract_target_feature("check the checkout rules", repo_id, chunks) == "checkout"


def test_a_query_naming_nothing_yields_no_feature(themed_repo):
    _, repo_id, chunks = themed_repo
    assert Agent._extract_target_feature("hello there", repo_id, chunks) == ""
    assert Agent._extract_target_feature("what is going on", repo_id, chunks) == ""


def test_structural_words_are_not_features(themed_repo):
    """`services`, `models`, `api` are how code is arranged, not what it does."""
    _, repo_id, chunks = themed_repo
    vocabulary = Agent._build_feature_vocabulary(repo_id, chunks)
    assert "services" not in vocabulary
    assert "models" not in vocabulary


def test_short_tokens_are_never_features(themed_repo):
    _, repo_id, chunks = themed_repo
    vocabulary = Agent._build_feature_vocabulary(repo_id, chunks)
    assert all(len(token) >= 4 for token in vocabulary)


# ------------------------------------------------------------------ the downstream effect

def test_a_bad_feature_no_longer_seeds_the_tracer(scanner_repo):
    """
    The reason this is more than cosmetic: the feature term seeds `FeatureTracer`, so a fragment
    produces a flow trace for something that does not exist.
    """
    from app.graph.feature_tracer import FeatureTracer

    _, repo_id, chunks = scanner_repo
    feature = Agent._extract_target_feature("what is the point of this?", repo_id, chunks)
    assert feature == ""

    # And the term that *is* extracted traces to something real.
    real = Agent._extract_target_feature("how does the scanner work?", repo_id, chunks)
    flow = FeatureTracer.trace_feature_flow(real, chunks)
    assert flow.get("feature") == real
