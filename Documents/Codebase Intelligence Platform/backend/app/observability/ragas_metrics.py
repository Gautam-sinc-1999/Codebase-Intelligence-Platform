"""
RAGAS-style metrics, computed with a judge model.

**These are faithful implementations of RAGAS's published metric definitions, not the `ragas`
package's output.** The library was not adopted deliberately: it pulls LangChain, `datasets` and
pandas into a codebase that has none of them, and wraps LLM access in its own configuration layer
that would sit awkwardly beside the `JUDGE_*` provider this system already needs. The metrics
themselves are a prompt and an arithmetic definition each, and implementing them directly keeps
the judge configurable the same way every other provider here is. Numbers will therefore be close
to, but not bit-identical with, the library's.

The five implemented are the ones the plan found actually applicable to a codebase assistant:

| Metric | Measures |
|---|---|
| `faithfulness` | Every claim in the answer is supported by the context it was given |
| `answer_relevancy` | The answer addresses the question that was asked (embedding similarity) |
| `context_precision` | The chunks that were retrieved are relevant, and ranked high |
| `context_entity_recall` | The symbols that matter appear in the retrieved context |
| `noise_sensitivity` | Irrelevant context did not corrupt the answer |

Context recall, answer correctness and semantic similarity were excluded: the first is superseded
by `graph_recall` for the queries that matter, and the last two need reference answers, which are
expensive to author for code and weak signals once written.

Every metric returns `None` rather than a number when it cannot be computed. A metric that
defaults to zero on a judge timeout reports a regression that did not happen.
"""
import logging
import math
from typing import Any, Dict, List, Optional

import httpx

from app.observability.judge import Judge, judge_configured

logger = logging.getLogger("observability.ragas")

# The metric names pushed as scores. Kept in one place so the harness, the docs and the dashboard
# cannot drift apart.
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_entity_recall",
    "noise_sensitivity",
]


def _context_block(contexts: List[str], limit: int = 12) -> str:
    """The retrieved context as the judge sees it, numbered so it can refer to chunks by rank."""
    return "\n\n".join(
        f"[{i + 1}] {text.strip()}" for i, text in enumerate(contexts[:limit]) if text
    )


# ------------------------------------------------------------------ faithfulness

FAITHFULNESS_PROMPT = """\
Break the ANSWER into individual factual claims about the code. For each claim, decide whether the
CONTEXT supports it. A claim is supported only if the context states or directly implies it —
plausible-sounding claims that the context does not establish are NOT supported.

Return JSON: {{"claims": [{{"claim": "...", "supported": true|false, "why": "..."}}]}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


async def faithfulness(judge: Judge, client: httpx.AsyncClient, *, question: str, answer: str,
                       contexts: List[str], **_kwargs) -> Optional[Dict[str, Any]]:
    """
    The fraction of the answer's claims that the retrieved context supports.

    The metric that catches invented behaviour — a confident description of a function that does
    something else, which is the failure mode a code assistant is most punished for.
    """
    if not answer.strip() or not contexts:
        return None

    parsed = await judge.ask_json(client, FAITHFULNESS_PROMPT.format(
        question=question, context=_context_block(contexts), answer=answer,
    ))
    claims = (parsed or {}).get("claims") if isinstance(parsed, dict) else None
    if not claims:
        return None

    supported = [c for c in claims if c.get("supported") is True]
    unsupported = [c for c in claims if c.get("supported") is not True]
    return {
        "value": len(supported) / len(claims),
        "comment": (f"{len(supported)}/{len(claims)} claims supported"
                    + (f"; unsupported: {unsupported[0].get('claim', '')[:90]}"
                       if unsupported else "")),
        "detail": {"claims": len(claims), "unsupported": len(unsupported)},
    }


# ------------------------------------------------------------------ answer relevancy

RELEVANCY_PROMPT = """\
Read the ANSWER only. Write the {n} questions it most directly and completely answers. Write them
as a user would ask them, without referring to "the answer".

Also decide whether the answer is **noncommittal** — evasive or a refusal, such as "I don't know",
"the context does not say", or a restatement of the question. An answer that declines to answer is
noncommittal even if it is polite and well written.

Return JSON: {{"generated_questions": ["...", "...", "..."], "noncommittal": true|false}}

ANSWER:
{answer}
"""


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _default_embedder(texts: List[str]) -> List[List[float]]:
    from app.vector.chromadb_client import ChromaVectorStore

    return ChromaVectorStore.embed_texts(texts)


async def answer_relevancy(judge: Judge, client: httpx.AsyncClient, *, question: str,
                           answer: str, n: int = 3, embedder=None,
                           **_kwargs) -> Optional[Dict[str, Any]]:
    """
    Whether the answer addresses the question that was asked.

    Follows RAGAS's actual method: generate questions from the answer alone, embed them alongside
    the original, and score by mean cosine similarity. The judge is deliberately **not** asked for
    a number — LLM-reported 0-1 scores are poorly calibrated and cluster on round values, so a real
    change of a few points is indistinguishable from the model rounding differently. Cosine
    similarity over a fixed embedding model is continuous and reproducible, which is the whole
    requirement for a regression check.

    A noncommittal answer scores 0 regardless of similarity. "The context does not say which
    function handles this" can be highly similar to the question while answering nothing.

    Uses the same all-MiniLM-L6-v2 the vector store already has loaded — a general-purpose
    sentence-similarity model, which is what this needs.
    """
    if not answer.strip() or not question.strip():
        return None

    parsed = await judge.ask_json(
        client, RELEVANCY_PROMPT.format(answer=answer, n=n)
    )
    if not isinstance(parsed, dict):
        return None

    if parsed.get("noncommittal") is True:
        return {
            "value": 0.0,
            "comment": "answer was noncommittal — it declined to answer the question",
            "detail": {"noncommittal": True, "generated_questions": []},
        }

    generated = [q for q in (parsed.get("generated_questions") or [])
                 if isinstance(q, str) and q.strip()]
    if not generated:
        return None

    embed = embedder or _default_embedder
    try:
        vectors = embed([question] + generated)
    except Exception as e:
        # Embedding is local, so this is a genuine fault rather than a rate limit — but it still
        # must not invent a score.
        logger.warning("Could not embed for answer relevancy: %s", e)
        return None

    if len(vectors) != len(generated) + 1:
        return None

    original, rest = vectors[0], vectors[1:]
    # Clamped at zero: cosine can go negative, and a negative relevance is not a meaningful
    # quantity to average across a dataset.
    similarities = [max(0.0, _cosine(original, v)) for v in rest]
    value = sum(similarities) / len(similarities)

    return {
        "value": max(0.0, min(1.0, value)),
        "comment": f"mean cosine over {len(generated)} generated question(s): "
                   + ", ".join(f"{s:.2f}" for s in similarities),
        "detail": {"generated_questions": generated, "similarities": similarities},
    }


# ------------------------------------------------------------------ context precision

PRECISION_PROMPT = """\
For each numbered CONTEXT chunk, decide whether it is useful for answering the QUESTION.

Return JSON: {{"verdicts": [{{"rank": 1, "useful": true|false}}, ...]}}

QUESTION:
{question}

CONTEXT:
{context}
"""


async def context_precision(judge: Judge, client: httpx.AsyncClient, *, question: str,
                            contexts: List[str], **_kwargs) -> Optional[Dict[str, Any]]:
    """
    Whether the retrieved chunks are relevant **and ranked high**.

    Rank-weighted on purpose: this is the metric that scores the fusion weights and the reranker.
    A run where the useful chunk is retrieved but sits at position six is a worse run than one
    where it sits at position one, and an unweighted mean cannot tell them apart.
    """
    if not contexts:
        return None

    parsed = await judge.ask_json(
        client, PRECISION_PROMPT.format(question=question, context=_context_block(contexts))
    )
    verdicts = (parsed or {}).get("verdicts") if isinstance(parsed, dict) else None
    if not verdicts:
        return None

    useful_at = {int(v["rank"]): bool(v.get("useful")) for v in verdicts
                 if isinstance(v, dict) and str(v.get("rank", "")).isdigit()}
    if not useful_at:
        return None

    # Average precision: precision@k at each rank where a useful chunk appears.
    hits = 0
    precision_sum = 0.0
    for rank in sorted(useful_at):
        if useful_at[rank]:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return {"value": 0.0, "comment": "no retrieved chunk was judged useful",
                "detail": {"useful": 0, "chunks": len(useful_at)}}

    return {
        "value": precision_sum / hits,
        "comment": f"{hits}/{len(useful_at)} chunks useful, rank-weighted",
        "detail": {"useful": hits, "chunks": len(useful_at)},
    }


# ------------------------------------------------------------------ context entity recall

async def context_entity_recall(judge: Judge, client: httpx.AsyncClient, *,
                                contexts: List[str], expected_entities: List[str],
                                **_kwargs) -> Optional[Dict[str, Any]]:
    """
    How many of the symbols that matter appear in the retrieved context.

    Computed **without** the judge. RAGAS extracts entities with an LLM because its entities are
    open-ended nouns; here they are symbol names the dataset already states as ground truth, so
    asking a model to find them would add cost, latency and a chance of being wrong to a string
    search that is exact.
    """
    if not expected_entities:
        return None

    from app.observability import scorers

    blob = "\n".join(contexts)
    found = [e for e in expected_entities if scorers._mentions(blob, e)]
    missing = [e for e in expected_entities if e not in found]
    return {
        "value": len(found) / len(expected_entities),
        "comment": (f"{len(found)}/{len(expected_entities)} expected symbols in context"
                    + (f"; missing {missing[:4]}" if missing else "")),
        "detail": {"missing": missing},
    }


# ------------------------------------------------------------------ noise sensitivity

NOISE_PROMPT = """\
Some CONTEXT chunks are irrelevant to the QUESTION. Decide whether the ANSWER was corrupted by
them — that is, whether it makes claims that come from an irrelevant chunk rather than a relevant
one.

Return JSON:
{{"claims_from_irrelevant_context": 0, "total_claims": 0, "examples": ["..."], "why": "..."}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


async def noise_sensitivity(judge: Judge, client: httpx.AsyncClient, *, question: str,
                            answer: str, contexts: List[str], **_kwargs) -> Optional[Dict[str, Any]]:
    """
    Whether irrelevant context corrupted the answer. **Lower is better.**

    Directly relevant here because two different things are injected into the prompt — graph facts
    and retrieved snippets — and the failure this catches is the model preferring a plausible
    snippet over the authoritative fact block.
    """
    if not answer.strip() or not contexts:
        return None

    parsed = await judge.ask_json(client, NOISE_PROMPT.format(
        question=question, context=_context_block(contexts), answer=answer,
    ))
    if not isinstance(parsed, dict):
        return None

    total = parsed.get("total_claims")
    corrupted = parsed.get("claims_from_irrelevant_context")
    if not isinstance(total, (int, float)) or not isinstance(corrupted, (int, float)) or total <= 0:
        return None

    return {
        "value": max(0.0, min(1.0, float(corrupted) / float(total))),
        "comment": f"{int(corrupted)}/{int(total)} claims traced to irrelevant context"
                   + (f" — {str(parsed.get('why', ''))[:120]}" if corrupted else ""),
        "detail": {"examples": parsed.get("examples", [])},
        # Named so a dashboard does not read a high number as a good one.
        "lower_is_better": True,
    }


# ------------------------------------------------------------------ the whole set

METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_entity_recall": context_entity_recall,
    "noise_sensitivity": noise_sensitivity,
}


async def evaluate_answer(judge: Judge, client: httpx.AsyncClient, *, question: str, answer: str,
                          contexts: List[str], expected_entities: Optional[List[str]] = None,
                          only: Optional[List[str]] = None,
                          embedder=None) -> Dict[str, Dict[str, Any]]:
    """
    Runs the metrics over one answered item.

    Metrics run sequentially rather than gathered, because they share one rate-limited judge and
    firing five at once is the quickest way to be throttled. A metric that fails is omitted, not
    zeroed — the harness reports what it could measure, and says what it could not.
    """
    if not judge_configured():
        return {}

    wanted = only or METRIC_NAMES
    results: Dict[str, Dict[str, Any]] = {}

    for name in wanted:
        metric = METRICS.get(name)
        if metric is None:
            continue
        try:
            outcome = await metric(
                judge, client,
                question=question, answer=answer, contexts=contexts,
                expected_entities=expected_entities or [],
                embedder=embedder,
            )
        except Exception as e:
            logger.warning("Metric '%s' failed: %s", name, e)
            outcome = None

        if outcome is not None:
            results[name] = outcome

    return results
