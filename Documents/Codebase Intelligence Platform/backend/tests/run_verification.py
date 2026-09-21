import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.discovery import FileDiscovery
from app.ingestion.ast_parser import MultiLanguageASTParser
from app.ingestion.chunker import HierarchicalCodeChunker
from app.vector.chromadb_client import chroma_store
from app.graph.neo4j_client import neo4j_client
from app.retrieval.intent_classifier import QueryIntentClassifier
from app.agents.impact_analyzer import ChangeImpactAnalyzer
from app.memory.conversation_memory import ConversationMemory

def run_tests():
    print("==================================================")
    print("RUNNING CODEBASE INTELLIGENCE VERIFICATION TESTS")
    print("==================================================")

    # Test 1: Multi-language AST Parsing & Line Number Preservation
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
    assert len(entities) >= 2, f"Expected >=2 entities, got {len(entities)}"
    
    class_ent = next(e for e in entities if e.entity_type == "class")
    assert class_ent.symbol == "PaymentService", f"Unexpected class symbol {class_ent.symbol}"
    assert class_ent.start_line == 2, f"Unexpected start line {class_ent.start_line}"

    func_ent = next(e for e in entities if e.entity_type == "method")
    assert func_ent.symbol == "PaymentService.calculate_total", f"Unexpected method symbol {func_ent.symbol}"
    assert func_ent.start_line == 3, f"Unexpected start line {func_ent.start_line}"
    assert func_ent.end_line == 5, f"Unexpected end line {func_ent.end_line}"
    print("✅ TEST 1 PASSED: AST Parsing & Exact Line Numbers Verified.")

    # Test 2: Query Intent Classifier
    loc_intent = QueryIntentClassifier.classify("Where is authentication implemented?")["intent"]
    assert loc_intent == "CODE_LOCATION", f"Got {loc_intent}"

    feat_intent = QueryIntentClassifier.classify("How does checkout work?")["intent"]
    assert feat_intent == "FEATURE_EXPLANATION", f"Got {feat_intent}"

    dep_intent = QueryIntentClassifier.classify("What depends on calculate_discount?")["intent"]
    assert dep_intent == "DEPENDENCY_ANALYSIS", f"Got {dep_intent}"

    impact_intent = QueryIntentClassifier.classify("If I change discount calculation, what will break?")["intent"]
    assert impact_intent == "CHANGE_IMPACT", f"Got {impact_intent}"
    print("✅ TEST 2 PASSED: Query Intent Classifier Verified.")

    # Test 3: ChromaDB Vector Indexing & Similarity Search
    repo_id = "test_indexing_repo"
    file_meta = {
        "repository_id": repo_id,
        "relative_path": "api/checkout.py",
        "language": "python",
        "content": "def process_checkout():\n    return 'done'\n"
    }

    entities = MultiLanguageASTParser.parse_file(file_meta)
    chunks = HierarchicalCodeChunker.create_chunks(file_meta, entities)

    chroma_store.add_chunks(repo_id, chunks)
    vector_res = chroma_store.search_similar(repo_id, "process_checkout", top_k=1)
    assert len(vector_res) > 0, "ChromaDB vector search yielded 0 results"
    print("✅ TEST 3 PASSED: ChromaDB Vector Store & Embedding Indexing Verified.")

    # Test 4: Neo4j Graph Builder & Graph Traversal Engine
    neo4j_client.build_repository_graph(repo_id, chunks)
    graph_data = neo4j_client.get_visual_graph_data(repo_id)
    assert len(graph_data["nodes"]) > 0, "Neo4j graph nodes were empty"
    print("✅ TEST 4 PASSED: Neo4j Cypher Graph Database & Nodes Verified.")

    # Test 5: MongoDB Conversation Memory & Working Context
    async def test_mongo():
        conv_id = "test_conv_999"
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
        assert len(history) >= 2, f"Expected history >= 2, got {len(history)}"

    asyncio.run(test_mongo())
    print("✅ TEST 5 PASSED: MongoDB Persistent Conversation Memory & History Resume Verified.")

    print("==================================================")
    print("ALL 5 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
