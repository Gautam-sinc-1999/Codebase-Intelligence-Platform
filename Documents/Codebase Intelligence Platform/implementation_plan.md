# Codebase Intelligence & Conversational RAG Platform

Architecture and implementation plan for a production-oriented Codebase Intelligence platform designed to onboard developers, explain complex execution flows, analyze code dependencies, and calculate change impacts across multi-language repositories.

---

## Technical Architecture Overview

The system uses a **Code-Aware RAG Architecture** that combines AST-based code parsing, hybrid ChromaDB vector/BM25 retrieval, Neo4j knowledge graph traversal, intent classification, and stateful agentic memory backed by MongoDB.

```text
                                +-----------------------------------+
                                |    React + Vite Chat & Graph UI   |
                                +-----------------+-----------------+
                                                  |
                                                  v
                                +-----------------+-----------------+
                                |      FastAPI Backend Gateway      |
                                +-----------------+-----------------+
                                                  |
                                                  v
                                +-----------------+-----------------+
                                |      LangGraph Agent Orchestration|
                                +-----------------+-----------------+
                                                  |
                 +--------------------------------+--------------------------------+
                 |                                |                                |
                 v                                v                                v
       +---------+--------+             +---------+--------+             +---------+--------+
       | Query Intent     |             | Memory &         |             | Repository       |
       | Router & Planner |             | Summarizer       |             | Context Manager  |
       +---------+--------+             +---------+--------+             +---------+--------+
                 |                                |                                |
                 v                                v                                v
       +---------+--------+             +---------+--------+             +---------+--------+
       | Hybrid Search    |             | Redis / MongoDB  |             | AST Code Chunks  |
       | (ChromaDB + BM25)|             | Persistence      |             | & Neo4j Graph   |
       +---------+--------+             +------------------+             | Database         |
                 |                                                         +------------------+
                 +--------------------------------+--------------------------------+
                                                  |
                                                  v
                                +-----------------+-----------------+
                                |   Hierarchical Context Builder    |
                                +-----------------+-----------------+
                                                  |
                                                  v
                                +-----------------+-----------------+
                                | LLM Reasoning & Guardrail Engine  |
                                +-----------------------------------+
```

---

## User Review Required

> [!IMPORTANT]
> **Updated Technology Stack & Database Selection**:
> 1. **Graph Database**: **Neo4j** (`neo4j` Python driver + Cypher query engine) for storing and traversing repository nodes (`File`, `Function`, `Class`, `API`, `DatabaseTable`) and relationships (`CALLS`, `IMPORTS`, `ROUTES_TO`, `READS_FROM`, `WRITES_TO`, `TESTS`).
> 2. **Vector Database**: **ChromaDB** (`chromadb` persistent & in-memory collection client) for vector embeddings and similarity search.
> 3. **Primary Database**: **MongoDB** (`pymongo` / `motor` async MongoDB driver) for persistent storage of repositories, files, metadata, conversations, working contexts, and chat history (replacing SQLite).
> 4. **Code-Aware Multi-Language Parser**: Tree-sitter / AST syntax parsers supporting Python, Java, JavaScript, TypeScript, React JSX/TSX, SQL, YAML, and Markdown.
> 5. **Configurable LLM & Embeddings Engine**: Support for OpenAI API keys, local embedding transformers, and fallback local mock generators so the platform can run seamlessly out-of-the-box.
> 6. **Sample Repository Included**: A pre-indexed e-commerce microservices repo (`backend/`, `frontend/`, `models/`, `tests/`) will be included for instant out-of-the-box testing.

---

## Open Questions

> [!NOTE]
> None at this stage. All technology specifications (Neo4j, ChromaDB, MongoDB) are incorporated.

---

## Proposed Changes

### 1. Project Directory Structure

```text
Codebase Intelligence Platform/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── repositories.py       # Upload, status, reindex endpoints
│   │   │   ├── conversations.py      # Conversation management & streaming chat
│   │   │   └── graph.py              # Dependency visualizer endpoints
│   │   ├── core/
│   │   │   ├── config.py             # App settings (Neo4j, ChromaDB, MongoDB, LLM options)
│   │   │   └── database.py           # MongoDB async client connection manager
│   │   ├── ingestion/
│   │   │   ├── discovery.py          # File walker & ignore pattern filter (.git, node_modules)
│   │   │   ├── ast_parser.py         # Multi-language Tree-sitter & AST code entity extractor
│   │   │   └── chunker.py            # Hierarchical symbol-aware code chunker
│   │   ├── graph/
│   │   │   ├── neo4j_client.py       # Neo4j connection & Cypher graph traversal service
│   │   │   └── feature_tracer.py     # End-to-end feature execution path tracer
│   │   ├── vector/
│   │   │   └── chromadb_client.py    # ChromaDB collection management & embedding store
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py   # ChromaDB + BM25 + Symbol + Neo4j hybrid search
│   │   │   ├── intent_classifier.py  # Intent detector (CODE_LOCATION, CHANGE_IMPACT, etc.)
│   │   │   └── reranker.py           # Cross-encoder / scoring reranker
│   │   ├── memory/
│   │   │   ├── conversation_memory.py# MongoDB persistent chat history & working memory
│   │   │   └── summarizer.py         # Incremental conversation summarizer
│   │   ├── agents/
│   │   │   ├── langgraph_agent.py    # LangGraph agent orchestration workflow
│   │   │   └── impact_analyzer.py    # Change impact & test coverage analyzer via Neo4j
│   │   └── main.py                   # FastAPI server entry point
│   ├── tests/                        # Backend unit & integration tests
│   └── requirements.txt
├── frontend/                         # Vite + React Modern Web UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx           # Repo selector & persistent chat history list
│   │   │   ├── ChatWindow.jsx        # Conversational UI with streaming responses
│   │   │   ├── MessageItem.jsx       # Markdown, code snippets, line numbers
│   │   │   ├── ExecutionFlow.jsx     # Visual step-by-step feature diagram
│   │   │   ├── DependencyGraph.jsx   # Interactive graph visualization (Vis.js / SVG)
│   │   │   ├── ImpactAnalysisCard.jsx# Confirmed vs Inferred impact analysis view
│   │   │   └── RepoUploader.jsx      # Zip / directory dropzone uploader
│   │   ├── services/
│   │   │   └── api.js                # REST & SSE WebSocket API service
│   │   ├── App.jsx                   # Main layout container
│   │   └── index.css                 # Glassmorphism dark mode aesthetic tokens
│   ├── package.json
│   └── vite.config.js
└── sample_repo/                      # Sample e-commerce application for out-of-the-box demo
    ├── backend/
    │   ├── api/checkout_controller.py
    │   ├── services/payment_service.py
    │   ├── services/discount_service.py
    │   └── models/order.py
    ├── frontend/
    │   ├── components/Checkout.jsx
    │   └── services/checkoutApi.js
    ├── database/schema.sql
    └── tests/test_discount.py
```

---

### Backend Component Breakdown

#### [NEW] [backend/requirements.txt](file:///Users/gautamkoshta/Documents/Codebase%20Intelligence%20Platform/backend/requirements.txt)
- Dependencies: `fastapi`, `uvicorn`, `pydantic`, `neo4j`, `chromadb`, `pymongo`, `motor`, `rank-bm25`, `sentence-transformers`, `httpx`, `python-multipart`, `tree-sitter`.

#### [NEW] [backend/app/core/database.py](file:///Users/gautamkoshta/Documents/Codebase%20Intelligence%20Platform/backend/app/core/database.py)
- MongoDB Connection Manager (`motor.motor_asyncio.AsyncIOMotorClient` or `pymongo.MongoClient` fallback), managing `repositories`, `files`, `conversations`, and `messages` collections.

#### [NEW] [backend/app/graph/neo4j_client.py](file:///Users/gautamkoshta/Documents/Codebase%20Intelligence%20Platform/backend/app/graph/neo4j_client.py)
- Neo4j Graph client driver for executing Cypher queries to build nodes (`File`, `Symbol`, `API`, `DatabaseTable`) and relationships (`CALLS`, `IMPORTS`, `ROUTES_TO`, `TESTS`), and performing multi-hop reverse dependency/impact traversals. Includes an embedded in-memory Neo4j fallback driver for self-contained local testing when a live Neo4j instance is not running.

#### [NEW] [backend/app/vector/chromadb_client.py](file:///Users/gautamkoshta/Documents/Codebase%20Intelligence%20Platform/backend/app/vector/chromadb_client.py)
- ChromaDB client managing vector collections per `repository_id` with semantic embedding indexing and similarity filtering.

#### [NEW] [backend/app/retrieval/hybrid_retriever.py](file:///Users/gautamkoshta/Documents/Codebase%20Intelligence%20Platform/backend/app/retrieval/hybrid_retriever.py)
- Hybrid RAG index combining ChromaDB similarity, BM25 keyword matching, exact symbol lookup, and Neo4j graph neighborhood expansion.

#### [NEW] [backend/app/memory/conversation_memory.py](file:///Users/gautamkoshta/Documents/Codebase%20Intelligence%20Platform/backend/app/memory/conversation_memory.py)
- MongoDB persistent conversation history store + working memory context (active feature, active symbols, recent turns, conversation summary).

---

## Verification Plan

### Automated Verification
1. **Backend Unit & Integration Tests**:
   - `python -m pytest backend/tests/` to verify AST parsing, ChromaDB vector indexing, Neo4j graph edge creation and Cypher queries, MongoDB conversation turn persistence, hybrid retrieval precision, and change impact calculations.
2. **API Endpoint Verification**:
   - Test upload endpoint `POST /repositories/upload` with sample repository.
   - Test conversation message endpoint `POST /conversations/{id}/messages` with sample queries:
     - "How does checkout work?" (Feature Tracing)
     - "Where is the discount calculation?" (Code Location)
     - "What depends on calculate_discount?" (Dependency Analysis via Neo4j)
     - "If I change discount calculation, what will be affected?" (Change Impact via Neo4j & ChromaDB)

### Manual Verification
1. Open the UI in browser.
2. Verify persistent chat history stored in MongoDB and resume conversation upon selecting a chat turn.
3. Verify interactive dependency graph powered by Neo4j graph relationships.
4. Test code snippet viewer for accurate file paths and line ranges.
