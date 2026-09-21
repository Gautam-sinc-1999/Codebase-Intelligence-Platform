"""Retrieval, ranking and query understanding. Covers F-07, F-19, F-20, F-32, F-38."""
import pytest

from app.retrieval.hybrid_retriever import HybridRetriever, BM25IndexCache
from app.retrieval.reranker import CodeReranker
from app.retrieval.intent_classifier import QueryIntentClassifier
from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent


# --------------------------------------------------------------------- intent

@pytest.mark.parametrize("query,intent", [
    ("Where is authentication implemented?", "CODE_LOCATION"),
    ("What depends on calculate_discount?", "DEPENDENCY_ANALYSIS"),
    ("If I change the discount logic what breaks?", "CHANGE_IMPACT"),
    ("How does checkout work?", "FEATURE_EXPLANATION"),
])
def test_intent_classification(query, intent):
    assert QueryIntentClassifier.classify(query)["intent"] == intent


# --------------------------------------------------------------------- symbol targeting (F-07)

@pytest.mark.parametrize("query,expected", [
    ("What depends on calculate_discount?", "DiscountService.calculate_discount"),
    ("Who calls `validate_coupon`?", "DiscountService.validate_coupon"),
    ("If I change process_checkout what breaks?", "process_checkout"),
    ("What does DiscountService do?", "DiscountService"),
    ("How does the app work?", ""),  # names nothing -> caller falls back to retrieval
])
def test_target_symbol_comes_from_the_query(sample_chunks, query, expected):
    assert Agent._extract_target_symbol(query, sample_chunks) == expected


def test_partial_matches_are_weighted_by_coverage(sample_chunks):
    """"discount" must bind to DiscountService, not to test_loyalty_tier_discount."""
    symbol = Agent._extract_target_symbol("Where is the discount calculation?", sample_chunks)
    assert "discount" in symbol.lower()
    assert not symbol.lower().startswith("test_")


def test_followup_references_are_recognised():
    assert Agent._is_followup_reference("Who calls it?")
    assert Agent._is_followup_reference("show me its callers")
    assert not Agent._is_followup_reference("Where is the payment service?")


# --------------------------------------------------------------------- feature vocabulary (F-20)

def test_feature_vocabulary_is_derived_from_the_repository(sample_chunks):
    assert Agent._extract_target_feature("How does checkout work?", "test_sample", sample_chunks) == "checkout"
    assert Agent._extract_target_feature("Explain the payment feature", "test_sample", sample_chunks) == "payment"


def test_feature_detection_works_for_any_domain(temp_repo):
    """A hardcoded six-word list meant only the bundled sample repo ever worked."""
    _, repo_id, chunks = temp_repo({
        "webhook_handler.py": "def handle_expense(payload):\n    return parse_expense(payload)\n",
        "expense_service.py": "def parse_expense(raw):\n    return raw\n",
    })
    assert Agent._extract_target_feature("How does expense tracking work?", repo_id, chunks) == "expense"
    assert Agent._extract_target_feature("Explain the webhook flow", repo_id, chunks) == "webhook"


@pytest.mark.parametrize("query", [
    "Show me the services", "What about the models?",
    "Explain the backend controller", "How is the api structured?",
])
def test_structural_words_are_not_features(sample_chunks, query):
    assert Agent._extract_target_feature(query, "test_sample", sample_chunks) == ""


# --------------------------------------------------------------------- lexical search (F-38)

@pytest.mark.parametrize("query,expected_symbol", [
    ("What depends on calculate_discount?", "calculate_discount"),
    ("Where is validate_coupon?", "validate_coupon"),
    ("process_checkout!", "process_checkout"),
])
def test_punctuation_does_not_defeat_lexical_search(sample_chunks, query, expected_symbol):
    """Whitespace splitting left "calculate_discount?" attached to its question mark."""
    results = HybridRetriever.retrieve("test_sample", query, sample_chunks, top_k=6)
    assert any(expected_symbol in c["symbol"] for c in results), \
        f"{expected_symbol} should be retrieved for {query!r}"


def test_retrieval_survives_an_empty_vector_store(temp_repo):
    """Lexical search must carry results when semantic search returns nothing."""
    from app.vector.chromadb_client import chroma_store
    _, repo_id, chunks = temp_repo({
        "billing.py": "def calculate_invoice_total(items):\n    return sum(items)\n"
    })
    chroma_store.delete_chunks(repo_id, [c["chunk_id"] for c in chunks])
    results = HybridRetriever.retrieve(repo_id, "where is calculate_invoice_total?", chunks, 6)
    assert results, "retrieval returned nothing despite the symbol being present"


# --------------------------------------------------------------------- BM25 caching (F-19)

def test_bm25_index_is_built_once_not_per_query(sample_chunks, monkeypatch):
    import rank_bm25
    builds = []
    real = rank_bm25.BM25Okapi

    class Counting(real):
        def __init__(self, corpus, *args, **kwargs):
            builds.append(len(corpus))
            super().__init__(corpus, *args, **kwargs)

    monkeypatch.setattr(rank_bm25, "BM25Okapi", Counting)
    BM25IndexCache.invalidate("test_sample")
    for query in ["discount", "checkout", "payment", "order", "coupon"]:
        HybridRetriever.retrieve("test_sample", query, sample_chunks, 6)

    assert len(builds) == 1, f"index rebuilt {len(builds)} times for 5 queries"


def test_bm25_index_rebuilds_after_invalidation(sample_chunks):
    BM25IndexCache.invalidate("test_sample")
    assert "test_sample" not in BM25IndexCache._indexes
    HybridRetriever.retrieve("test_sample", "discount", sample_chunks, 6)
    assert "test_sample" in BM25IndexCache._indexes


# --------------------------------------------------------------------- reranking (F-32)

def _candidate(chunk_id, symbol, entity_type, path, snippet, start, end):
    return {"chunk_id": chunk_id, "symbol": symbol, "entity_type": entity_type,
            "file_path": path, "code_snippet": snippet, "start_line": start, "end_line": end}


CANDIDATES = [
    _candidate("1", "services/discount_service.py", "module", "services/discount_service.py", "x" * 3000, 1, 500),
    _candidate("2", "DiscountService.calculate_discount", "method", "services/discount_service.py",
               "def calculate_discount(self, subtotal):\n    return subtotal * rate", 15, 33),
    _candidate("3", "test_discount_applies", "function", "tests/test_discount.py",
               "def test_discount_applies():\n    assert calculate_discount(100)", 4, 7),
]
FUSED = {"1": 5.0, "2": 4.0, "3": 4.5}   # the whole-file chunk ranks highest before reranking


def test_definition_outranks_whole_file_chunk():
    ranked = CodeReranker.rerank(CANDIDATES, "where is calculate_discount implemented", FUSED, 3)
    assert ranked[0]["symbol"] == "DiscountService.calculate_discount"


def test_tests_are_promoted_only_when_asked_for():
    for_code = CodeReranker.rerank(CANDIDATES, "where is calculate_discount implemented", FUSED, 3)
    for_tests = CodeReranker.rerank(CANDIDATES, "which tests cover the discount", FUSED, 3)
    assert "test" not in for_code[0]["file_path"]
    assert "test" in for_tests[0]["file_path"]


def test_reranker_handles_empty_input():
    assert CodeReranker.rerank([], "anything", {}, 5) == []
