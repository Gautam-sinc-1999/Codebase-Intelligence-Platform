import sys
import os
import asyncio
import zipfile
import tempfile
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.discovery import FileDiscovery
from app.ingestion.ast_parser import MultiLanguageASTParser
from app.ingestion.chunker import HierarchicalCodeChunker
from app.vector.chromadb_client import chroma_store
from app.graph.neo4j_client import neo4j_client
from app.graph.feature_tracer import FeatureTracer
from app.agents.impact_analyzer import ChangeImpactAnalyzer
from app.retrieval.intent_classifier import QueryIntentClassifier
from app.retrieval.hybrid_retriever import HybridRetriever
from app.agents.langgraph_agent import CodebaseAgentOrchestrator
from app.memory.conversation_memory import ConversationMemory
from app.core.config import settings

async def run_full_e2e_pipeline():
    print("================================================================")
    print("🚀 STARTING FULL END-TO-END CODEBASE INTELLIGENCE PIPELINE TEST")
    print("================================================================")

    failures = []

    # Phase 1: Ingestion & Auto-Indexing Pipeline
    print("\n--- PHASE 1: Repository File Discovery, AST Parsing & Chunking ---")
    sample_dir = settings.SAMPLE_REPO_DIR
    repo_id = "sample_ecommerce_repo"

    try:
        discovered_files = FileDiscovery.discover_repository(sample_dir, repo_id)
        print(f"✅ Discovered {len(discovered_files)} files in repository.")
        assert len(discovered_files) >= 5, "Expected >= 5 repository files"

        all_chunks = []
        for f_meta in discovered_files:
            entities = MultiLanguageASTParser.parse_file(f_meta)
            chunks = HierarchicalCodeChunker.create_chunks(f_meta, entities)
            all_chunks.extend(chunks)

        print(f"✅ AST Syntax Parsing extracted {len(all_chunks)} symbol-aware code chunks.")
        assert len(all_chunks) >= 5, "Expected >= 5 AST code chunks"
    except Exception as e:
        failures.append(f"Phase 1 Ingestion failed: {e}")

    # Phase 2: Vector DB & Neo4j Graph Indexing
    print("\n--- PHASE 2: ChromaDB Vector Store & Neo4j Graph Indexing ---")
    try:
        chroma_store.add_chunks(repo_id, all_chunks)
        vector_res = chroma_store.search_similar(repo_id, "process_checkout", top_k=2)
        print(f"✅ ChromaDB Vector Search -> Returned {len(vector_res)} semantically matched chunks.")

        neo4j_client.build_repository_graph(repo_id, all_chunks)
        graph_data = neo4j_client.get_visual_graph_data(repo_id)
        print(f"✅ Neo4j Knowledge Graph -> Topology built with {len(graph_data['nodes'])} Nodes & {len(graph_data['edges'])} Relationships.")
    except Exception as e:
        failures.append(f"Phase 2 DB Indexing failed: {e}")

    # Phase 3: Conversational Agent & Intent Router
    print("\n--- PHASE 3: Agentic Execution Graph & Intent Routing ---")
    conv_id = "e2e_conv_1001"
    try:
        conv = await ConversationMemory.get_or_create_conversation(conv_id, repo_id, "E2E Pipeline Session")
        print(f"✅ Created conversation session '{conv['conversation_id']}' in MongoDB.")

        # Query 1: Feature Explanation Flow
        q1 = await CodebaseAgentOrchestrator.process_user_query(
            repository_id=repo_id,
            conversation_id=conv_id,
            query="How does checkout work?",
            all_chunks=all_chunks,
            existing_summary=conv.get("summary", ""),
            working_context=conv.get("working_context", {})
        )
        await ConversationMemory.save_turn(conv_id, "How does checkout work?", q1)
        print(f"✅ Q1 (Feature Flow) -> Intent: '{q1['intent']}', Execution Steps: {len(q1['execution_flow'].get('flow_steps', []))}, Sources: {len(q1['sources'])}")

        # Query 2: Code Location
        q2 = await CodebaseAgentOrchestrator.process_user_query(
            repository_id=repo_id,
            conversation_id=conv_id,
            query="Where is discount calculation?",
            all_chunks=all_chunks,
            existing_summary=q1["updated_summary"],
            working_context=q1["working_context"]
        )
        await ConversationMemory.save_turn(conv_id, "Where is discount calculation?", q2)
        print(f"✅ Q2 (Code Location) -> Intent: '{q2['intent']}', Symbol: '{q2['sources'][0]['symbol'] if q2['sources'] else 'N/A'}' ({q2['sources'][0]['file_path'] if q2['sources'] else 'N/A'})")

        # Query 3: Dependency Analysis via Neo4j
        q3 = await CodebaseAgentOrchestrator.process_user_query(
            repository_id=repo_id,
            conversation_id=conv_id,
            query="What depends on calculate_discount?",
            all_chunks=all_chunks,
            existing_summary=q2["updated_summary"],
            working_context=q2["working_context"]
        )
        await ConversationMemory.save_turn(conv_id, "What depends on calculate_discount?", q3)
        print(f"✅ Q3 (Dependency Tracing) -> Intent: '{q3['intent']}', Sources: {len(q3['sources'])}")

        # Query 4: Change Impact Analysis
        q4 = await CodebaseAgentOrchestrator.process_user_query(
            repository_id=repo_id,
            conversation_id=conv_id,
            query="If I change discount calculation, what will be affected?",
            all_chunks=all_chunks,
            existing_summary=q3["updated_summary"],
            working_context=q3["working_context"]
        )
        await ConversationMemory.save_turn(conv_id, "If I change discount calculation, what will be affected?", q4)
        impact = q4.get("impact_analysis", {})
        print(f"✅ Q4 (Change Impact) -> Intent: '{q4['intent']}', Primary Target: '{impact.get('primary_target', {}).get('symbol')}', Confirmed Callers: {len(impact.get('confirmed_callers', []))}, Affected Tests: {len(impact.get('affected_tests', []))}")

    except Exception as e:
        failures.append(f"Phase 3 Agentic Execution failed: {e}")

    # Phase 4: MongoDB Persistent History & Resume Chat
    print("\n--- PHASE 4: MongoDB Persistent Conversation History & Summary ---")
    try:
        history = await ConversationMemory.load_conversation_history(conv_id)
        conv_doc = await ConversationMemory.get_or_create_conversation(conv_id, repo_id)
        print(f"✅ Loaded persistent chat history ({len(history)} messages persisted in MongoDB).")
        print(f"✅ Conversation Summary: \"{conv_doc.get('summary', '')}\"")
        assert len(history) >= 8, f"Expected >= 8 messages in history, got {len(history)}"
    except Exception as e:
        failures.append(f"Phase 4 MongoDB history failed: {e}")

    # Phase 5: Dynamic Repository Zip Archive Ingestion Pipeline
    print("\n--- PHASE 5: Dynamic Zip Upload & Re-Indexing Pipeline ---")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        tmp_zip_path = tmp_zip.name

    try:
        with zipfile.ZipFile(tmp_zip_path, 'w') as zf:
            zf.writestr("services/auth_service.py", "def authenticate_user(username, password):\n    return True\n")
            zf.writestr("controllers/auth_controller.py", "from services.auth_service import authenticate_user\ndef login():\n    return authenticate_user('admin', 'secret')\n")
            zf.writestr("tests/test_auth.py", "from services.auth_service import authenticate_user\ndef test_auth():\n    assert authenticate_user('a', 'b') == True\n")

        dyn_repo_id = "dynamic_auth_repo"
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        dyn_files = FileDiscovery.discover_repository(temp_dir, dyn_repo_id)
        dyn_chunks = []
        for f in dyn_files:
            entities = MultiLanguageASTParser.parse_file(f)
            dyn_chunks.extend(HierarchicalCodeChunker.create_chunks(f, entities))

        chroma_store.add_chunks(dyn_repo_id, dyn_chunks)
        neo4j_client.build_repository_graph(dyn_repo_id, dyn_chunks)
        dyn_graph = neo4j_client.get_visual_graph_data(dyn_repo_id)

        print(f"✅ Dynamic Zip Upload Pipeline -> Indexed {len(dyn_files)} files, {len(dyn_chunks)} AST chunks, built Neo4j graph with {len(dyn_graph['nodes'])} nodes.")
    except Exception as e:
        failures.append(f"Phase 5 Zip upload pipeline failed: {e}")
    finally:
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)

    print("\n================================================================")
    if not failures:
        print("🎉 ALL END-TO-END PIPELINE STAGES COMPLETED WITH 100% SUCCESS!")
        print("================================================================")
    else:
        print("⚠️ END-TO-END PIPELINE ENCOUNTERED THE FOLLOWING ISSUES:")
        for f in failures:
            print(f"   ❌ {f}")
        print("================================================================")

if __name__ == "__main__":
    asyncio.run(run_full_e2e_pipeline())
