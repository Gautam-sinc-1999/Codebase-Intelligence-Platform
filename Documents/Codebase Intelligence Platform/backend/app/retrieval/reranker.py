import re
import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger("retrieval.reranker")


class CodeReranker:
    # Scores from the most recent rerank, keyed by chunk_id. Read by tracing; never written to
    # the chunks themselves, which are shared cache objects.
    last_scores: Dict[str, Dict[str, float]] = {}

    """
    Reorders fused retrieval candidates using signals specific to code.

    Ranking previously lived inline in the retriever as additive magic numbers (+2.5 vector,
    +3.0 symbol, +2.0 path, +1.5 graph), with no notion of *what kind* of result a candidate was.
    A file-level `module` stand-in and a precise function definition scored alike, so whole-file
    chunks — which carry the least information per token — routinely displaced the definition
    the developer asked for.

    Deliberately dependency-free. A cross-encoder was the original plan, but it would reintroduce
    `sentence-transformers` and `torch` (~2.5 GB, removed in F-28) to reorder a handful of
    candidates, and the signals below are the ones that actually separate code results.
    """

    # Entity kinds, weighted by how much a developer's question is usually *about* them.
    ENTITY_WEIGHTS = {
        "method": 1.00,
        "function": 1.00,
        "endpoint": 1.00,
        "component": 0.95,
        "class": 0.90,
        "table": 0.80,
        "module": 0.45,   # whole-file stand-in: real content, but nothing was located within it
    }

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on", "at", "to", "for",
        "and", "or", "how", "does", "do", "what", "where", "which", "who", "why", "when", "it",
        "its", "this", "that", "with", "from", "into", "about", "show", "me", "find", "explain",
        "code", "file", "files", "line", "lines", "work", "works", "used", "use",
    }

    @classmethod
    def _terms(cls, text: str) -> Set[str]:
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        return {
            token.lower()
            for token in re.split(r"[^A-Za-z0-9]+", spaced)
            if len(token) > 2 and token.lower() not in cls.STOPWORDS
        }

    @classmethod
    def score(cls, chunk: Dict[str, Any], query: str, fused_score: float) -> float:
        """Returns a final ranking score for one candidate."""
        query_terms = cls._terms(query)
        symbol = chunk.get("symbol", "") or ""
        symbol_terms = cls._terms(symbol)

        score = fused_score

        # 1. Entity kind. Multiplicative, so it shapes ranking without swamping retrieval signal.
        score *= cls.ENTITY_WEIGHTS.get(chunk.get("entity_type", ""), 0.7)

        # 2. Exact symbol match — the strongest evidence a developer named this thing.
        symbol_lower = symbol.lower()
        symbol_tail = symbol_lower.rsplit(".", 1)[-1]
        if query_terms:
            if symbol_lower in query.lower() or symbol_tail in query_terms:
                score += 4.0
            elif symbol_terms & query_terms:
                overlap = len(symbol_terms & query_terms) / len(symbol_terms | query_terms)
                score += 2.0 * overlap

        # 3. How much of the question the code body actually covers.
        if query_terms:
            snippet_terms = cls._terms(chunk.get("code_snippet", "")[:2000])
            coverage = len(query_terms & snippet_terms) / len(query_terms)
            score += 1.5 * coverage

        # 4. Path relevance, excluding the generic directory names every repo shares.
        path_terms = cls._terms(chunk.get("file_path", "")) - {
            "src", "app", "lib", "backend", "frontend", "api", "services", "components", "utils"
        }
        if query_terms & path_terms:
            score += 1.0

        # 5. Prefer definitions that are actually readable. Very small chunks are usually stubs;
        # very large ones are usually whole files that happened to match somewhere.
        span = max(1, int(chunk.get("end_line", 1)) - int(chunk.get("start_line", 1)) + 1)
        if span <= 2:
            score -= 0.5
        elif span > 400:
            score -= 1.0

        # 6. Tests answer "how is this verified", not "where does this live" — unless asked for.
        path_lower = (chunk.get("file_path", "") or "").lower()
        if ("test" in path_lower or "spec" in path_lower) and not (
            query_terms & {"test", "tests", "spec", "coverage", "verified"}
        ):
            score -= 1.0

        return score

    @classmethod
    def rerank(
        cls,
        candidates: List[Dict[str, Any]],
        query: str,
        fused_scores: Dict[str, float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top_k candidates in final ranked order.

        The scores that produced that order are recorded on `last_scores` rather than attached to
        the chunks. The chunks are the cached index objects, shared with every other query and with
        the stored citations — writing a per-query score onto them would leak it into both.
        """
        cls.last_scores = {}
        if not candidates:
            return []

        scored = [
            (cls.score(chunk, query, fused_scores.get(chunk["chunk_id"], 0.0)), chunk)
            for chunk in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:top_k]
        cls.last_scores = {
            chunk["chunk_id"]: {
                "rerank": round(score, 4),
                "fused": round(fused_scores.get(chunk["chunk_id"], 0.0), 4),
            }
            for score, chunk in top
        }
        return [chunk for _, chunk in top]
