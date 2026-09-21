import pytest
import asyncio
from app.ingestion.discovery import FileDiscovery
from app.ingestion.ast_parser import MultiLanguageASTParser
from app.ingestion.chunker import HierarchicalCodeChunker
from app.vector.chromadb_client import chroma_store
from app.graph.neo4j_client import neo4j_client
from app.retrieval.intent_classifier import QueryIntentClassifier
from app.retrieval.hybrid_retriever import HybridRetriever
from app.agents.impact_analyzer import ChangeImpactAnalyzer
from app.memory.conversation_memory import ConversationMemory

def test_ast_parsing_line_numbers():
    python_code = """
class PaymentService:
    def calculate_total(self, amount, discount):
        subtotal = amount - discount
        return subtotal
"""
    file_meta = {
        "repository_id": "test_repo",
        "relative_path": "services/payment.py",
        "language": "python",
        "content": python_code
    }

    entities = MultiLanguageASTParser.parse_file(file_meta)
    assert len(entities) >= 2
    
    class_ent = next(e for e in entities if e.entity_type == "class")
    assert class_ent.symbol == "PaymentService"
    assert class_ent.start_line == 2

    func_ent = next(e for e in entities if e.entity_type == "method")
    assert func_ent.symbol == "PaymentService.calculate_total"
    assert func_ent.start_line == 3
    assert func_ent.end_line == 5

def test_intent_classification():
    assert QueryIntentClassifier.classify("Where is authentication implemented?")["intent"] == "CODE_LOCATION"
    assert QueryIntentClassifier.classify("How does checkout work?")["intent"] == "FEATURE_EXPLANATION"
    assert QueryIntentClassifier.classify("What depends on calculate_discount?")["intent"] == "DEPENDENCY_ANALYSIS"
    assert QueryIntentClassifier.classify("If I change discount calculation, what will break?")["intent"] == "CHANGE_IMPACT"

def test_chromadb_and_neo4j_indexing():
    repo_id = "test_indexing_repo"
    file_meta = {
        "repository_id": repo_id,
        "relative_path": "api/checkout.py",
        "language": "python",
        "content": "def process_checkout():\n    return 'done'\n"
    }

    entities = MultiLanguageASTParser.parse_file(file_meta)
    chunks = HierarchicalCodeChunker.create_chunks(file_meta, entities)

    # Test ChromaDB indexing
    chroma_store.add_chunks(repo_id, chunks)
    results = chroma_store.search_similar(repo_id, "process_checkout", top_k=1)
    assert len(results) > 0

    # Test Neo4j indexing
    neo4j_client.build_repository_graph(repo_id, chunks)
    graph_data = neo4j_client.get_visual_graph_data(repo_id)
    assert len(graph_data["nodes"]) > 0

@pytest.mark.asyncio
async def test_mongodb_conversation_memory():
    conv_id = "test_conv_123"
    repo_id = "test_repo"

    conv = await ConversationMemory.get_or_create_conversation(conv_id, repo_id, "Test Title")
    assert conv["conversation_id"] == conv_id

    assistant_resp = {
        "answer": "Test answer response",
        "sources": [],
        "working_context": {"current_feature": "checkout"},
        "updated_summary": "Summary updated"
    }

    await ConversationMemory.save_turn(conv_id, "User question test", assistant_resp)
    history = await ConversationMemory.load_conversation_history(conv_id)
    assert len(history) >= 2
