import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.vector.chromadb_client import chroma_store
from app.graph.neo4j_client import neo4j_client
from app.retrieval.reranker import CodeReranker

logger = logging.getLogger("retrieval.hybrid")


class BM25IndexCache:
    """
    Keeps one built BM25 index per repository.

    The index was previously rebuilt inside every query: the entire corpus was re-tokenised and
    re-weighted per message, so keyword search cost grew linearly with repository size on each
    turn rather than once at indexing time.

    Entries are invalidated explicitly when a repository is re-indexed, with the chunk count as
    a cheap secondary guard against a stale index surviving a missed invalidation.
    """

    _indexes: Dict[str, Tuple[int, Any, List[str]]] = {}

    @classmethod
    def invalidate(cls, repository_id: str) -> None:
        cls._indexes.pop(repository_id, None)

    @classmethod
    def get(cls, repository_id: str, all_chunks: List[Dict[str, Any]]) -> Optional[Tuple[Any, List[str]]]:
        """Returns (bm25, chunk_ids) or None when rank_bm25 is unavailable."""
        cached = cls._indexes.get(repository_id)
        if cached and cached[0] == len(all_chunks):
            return cached[1], cached[2]

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return None

        corpus = [chunk["formatted_content"].lower().split() for chunk in all_chunks]
        if not corpus:
            return None

        bm25 = BM25Okapi(corpus)
        chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]
        cls._indexes[repository_id] = (len(all_chunks), bm25, chunk_ids)
        logger.info("Built BM25 index for '%s' over %d chunks.", repository_id, len(all_chunks))
        return bm25, chunk_ids


class HybridRetriever:
    """
    Hybrid code retrieval engine combining:
    - Semantic Search (ChromaDB)
    - Keyword Search (BM25 / TF-IDF Keyword Matcher)
    - Symbol Search (Exact match)
    - Graph Neighborhood Expansion (Neo4j)
    """

    @classmethod
    def retrieve(
        cls,
        repository_id: str,
        query: str,
        all_chunks: List[Dict[str, Any]],
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        if not all_chunks:
            return []

        candidates_map: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}

        # 1. ChromaDB Semantic Vector Search
        vector_results = chroma_store.search_similar(repository_id, query, top_k=top_k)
        for v_item in vector_results:
            chunk_id = v_item["chunk_id"]
            chunk = next((c for c in all_chunks if c["chunk_id"] == chunk_id), None)
            if chunk:
                candidates_map[chunk_id] = chunk
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 2.5

        # Query tokens are split on non-word characters rather than whitespace. Splitting on
        # whitespace left punctuation attached, so "calculate_discount?" never matched the symbol
        # `calculate_discount` — lexical search silently failed on ordinary questions and the
        # results leaned entirely on the vector store.
        query_tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if t]

        # 2. BM25 keyword search, against the index built once per repository.
        bm25_index = BM25IndexCache.get(repository_id, all_chunks)
        if bm25_index is not None:
            bm25, chunk_ids = bm25_index
            bm25_scores = bm25.get_scores(query_tokens)

            chunks_by_id = {chunk["chunk_id"]: chunk for chunk in all_chunks}
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
            for idx in top_bm25_indices:
                if bm25_scores[idx] > 0 and idx < len(chunk_ids):
                    # Look the chunk up by id rather than by position: the cached index and the
                    # caller's list are only guaranteed to agree on ids, not on ordering.
                    chunk = chunks_by_id.get(chunk_ids[idx])
                    if chunk:
                        cid = chunk["chunk_id"]
                        candidates_map[cid] = chunk
                        scores[cid] = scores.get(cid, 0.0) + (bm25_scores[idx] * 0.1)
        else:
            # Fallback keyword term frequency scorer
            query_words = set(query_tokens)
            for chunk in all_chunks:
                cid = chunk["chunk_id"]
                content = chunk["formatted_content"].lower()
                matches = sum(1 for w in query_words if w in content)
                if matches > 0:
                    candidates_map[cid] = chunk
                    scores[cid] = scores.get(cid, 0.0) + (matches * 0.5)

        # 3. Exact Symbol / Endpoint Match
        query_words = set(query_tokens)
        for chunk in all_chunks:
            symbol_lower = chunk["symbol"].lower()
            file_lower = chunk["file_path"].lower()
            cid = chunk["chunk_id"]

            if any(w in symbol_lower for w in query_words if len(w) > 3):
                candidates_map[cid] = chunk
                scores[cid] = scores.get(cid, 0.0) + 3.0

            if any(w in file_lower for w in query_words if len(w) > 3):
                candidates_map[cid] = chunk
                scores[cid] = scores.get(cid, 0.0) + 2.0

        # 4. Neo4j Graph Traversal for candidates
        symbols_to_expand = [chunk["symbol"] for chunk in candidates_map.values() if chunk.get("symbol")]
        graph_expanded_symbols = set()
        for sym in symbols_to_expand[:3]:
            forward_deps = neo4j_client.get_forward_dependencies(sym, max_depth=1, repository_id=repository_id)
            for dep in forward_deps:
                graph_expanded_symbols.add(dep.get("symbol", "").lower())

        for chunk in all_chunks:
            if chunk["symbol"].lower() in graph_expanded_symbols:
                cid = chunk["chunk_id"]
                candidates_map[cid] = chunk
                scores[cid] = scores.get(cid, 0.0) + 1.5

        # Rerank. The fused scores above say how many retrieval signals found a candidate; the
        # reranker decides which of those is the best *answer*, using code-specific signals.
        return CodeReranker.rerank(
            candidates=list(candidates_map.values()),
            query=query,
            fused_scores=scores,
            top_k=top_k,
        )
