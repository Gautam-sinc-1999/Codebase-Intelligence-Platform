import os
import logging
import math
from typing import List, Dict, Any

logger = logging.getLogger("chromadb")

class MemoryVectorStoreFallback:
    """
    In-memory vector store simulating vector similarity search and metadata filtering
    when chromadb package is not installed in local environment.
    """
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []

    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_texts: List[str], n_results: int = 5) -> Dict[str, Any]:
        if not self.documents:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        query_terms = set(query_texts[0].lower().split())

        scored_results = []
        for doc, meta, doc_id in zip(self.documents, self.metadatas, self.ids):
            doc_terms = set(doc.lower().split())
            intersection = query_terms.intersection(doc_terms)
            score = len(intersection) / max(1, math.sqrt(len(query_terms) * len(doc_terms)))
            scored_results.append((score, doc, meta, doc_id))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_items = scored_results[:n_results]

        return {
            "documents": [[item[1] for item in top_items]],
            "metadatas": [[item[2] for item in top_items]],
            "ids": [[item[3] for item in top_items]]
        }

    def count(self):
        return len(self.documents)

class ChromaVectorStore:
    """
    ChromaDB vector database service for semantic code retrieval.
    Includes persistent Client + in-memory similarity fallback.
    """

    def __init__(self):
        self.is_fallback = False
        self.collections = {}
        self.client = None
        try:
            import chromadb  # noqa: F401
        except ImportError:
            logger.warning("chromadb package not installed. Utilizing ChromaDB fallback store.")
            self.is_fallback = True
            return
        self._open_client()

    def _open_client(self):
        """Creates the underlying client. Separated from __init__ so it can be reopened."""
        import chromadb
        from app.core.config import settings
        try:
            os.makedirs(settings.CHROMADB_DIR, exist_ok=True)
            self.client = chromadb.PersistentClient(path=settings.CHROMADB_DIR)
        except Exception as e:
            logger.warning("Could not initialize persistent ChromaDB (%s). Using ephemeral Client.", e)
            self.client = chromadb.Client()

    def _ensure_client(self):
        """
        Returns a usable client, reopening it if it was closed.

        `close()` is called on application shutdown, but this store is a module-level singleton:
        without this, a second application lifecycle in the same process (tests, an embedded
        host, a reload) would inherit a dead client and every vector operation would fail.
        """
        if self.client is None or getattr(self.client, "_closed", False):
            self._open_client()
        return self.client

    # The embedding model is loaded once and kept. ChromaDB's DefaultEmbeddingFunction
    # constructs a fresh ONNXMiniLM_L6_V2 on *every* call, which reloads the model each time —
    # fine when it happens once per collection write, wasteful when embedding a handful of
    # strings in a loop.
    _embedder = None

    @classmethod
    def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        """
        Embeds arbitrary text with the same model the vector store uses (all-MiniLM-L6-v2, 384-d).

        Exposed because similarity between two pieces of *text* is useful outside retrieval —
        the answer-relevancy metric compares a question against questions generated from an
        answer, and doing that with the model already loaded costs nothing extra.
        """
        if not texts:
            return []
        if cls._embedder is None:
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
            cls._embedder = ONNXMiniLM_L6_V2()
        return [list(vector) for vector in cls._embedder(list(texts))]

    def get_or_create_collection(self, repository_id: str):
        if self.is_fallback:
            if repository_id not in self.collections:
                self.collections[repository_id] = MemoryVectorStoreFallback()
            return self.collections[repository_id]

        collection_name = f"repo_{repository_id.replace('-', '_')}"
        return self._ensure_client().get_or_create_collection(name=collection_name)

    def add_chunks(self, repository_id: str, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        collection = self.get_or_create_collection(repository_id)

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            doc_text = f"Symbol: {chunk['symbol']}\nPath: {chunk['file_path']}\nType: {chunk['entity_type']}\nSummary: {chunk['summary']}\nCode:\n{chunk['code_snippet']}"
            documents.append(doc_text)
            
            meta = {
                "repository_id": str(chunk["repository_id"]),
                "file_path": str(chunk["file_path"]),
                "language": str(chunk["language"]),
                "entity_type": str(chunk["entity_type"]),
                "symbol": str(chunk["symbol"]),
                "start_line": int(chunk["start_line"]),
                "end_line": int(chunk["end_line"]),
                "parent_symbol": str(chunk.get("parent_symbol") or ""),
                "summary": str(chunk["summary"])
            }
            metadatas.append(meta)
            ids.append(chunk["chunk_id"])

        if self.is_fallback:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
        else:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    def close(self):
        """
        Releases the ChromaDB client explicitly.

        Left to interpreter teardown, the client intermittently aborts the process with
        `recursive_mutex lock failed` (SIGABRT) — a race in its own shutdown path, reproducible
        roughly one run in three under heavy use. Closing deterministically on application
        shutdown avoids relying on that ordering. See F-36.
        """
        if self.is_fallback:
            return
        client = getattr(self, "client", None)
        if client is None:
            return
        try:
            client.close()
            logger.info("ChromaDB client closed.")
        except Exception as e:
            logger.warning("Error closing ChromaDB client: %s", e)
        finally:
            # Dropped rather than left dangling, so the next use reopens it (see _ensure_client).
            self.client = None

    def delete_collection(self, repository_id: str) -> bool:
        """
        Drops a repository's entire collection.

        Deleting the chunk ids one by one would leave the collection itself behind, and an empty
        collection is not free: it still occupies a directory in the persistent store and still
        appears in `list_collections()`. Repository deletion should leave nothing addressable.
        """
        try:
            if self.is_fallback:
                return self.collections.pop(repository_id, None) is not None

            collection_name = f"repo_{repository_id.replace('-', '_')}"
            self._ensure_client().delete_collection(name=collection_name)
            logger.info("Deleted vector collection for '%s'.", repository_id)
            return True
        except Exception as e:
            # Already absent is the expected case when indexing failed before the first write.
            logger.warning("Could not delete vector collection for '%s': %s", repository_id, e)
            return False

    def delete_chunks(self, repository_id: str, chunk_ids: List[str]):
        """Removes chunks whose source files no longer exist, so deleted code stops being retrieved."""
        if not chunk_ids:
            return
        try:
            collection = self.get_or_create_collection(repository_id)
            if self.is_fallback:
                keep = [i for i, cid in enumerate(collection.ids) if cid not in set(chunk_ids)]
                collection.documents = [collection.documents[i] for i in keep]
                collection.metadatas = [collection.metadatas[i] for i in keep]
                collection.ids = [collection.ids[i] for i in keep]
            else:
                collection.delete(ids=chunk_ids)
        except Exception as e:
            logger.error("Could not delete %d stale chunks for %s: %s", len(chunk_ids), repository_id, e)

    def search_similar(self, repository_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            collection = self.get_or_create_collection(repository_id)
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count() or 1)
            )

            matched_chunks = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                for doc, meta, doc_id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
                    matched_chunks.append({
                        "chunk_id": doc_id,
                        "document": doc,
                        "metadata": meta,
                        "score": 0.85
                    })
            return matched_chunks
        except Exception as e:
            logger.error("ChromaDB query error for repo %s: %s", repository_id, e)
            return []

chroma_store = ChromaVectorStore()
