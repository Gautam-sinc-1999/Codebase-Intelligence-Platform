"""
Deterministic answer scores — no LLM judge, no labelled data, no added latency.

Every metric here is computed from artefacts the system already produces: the chunk index, the
call graph, and the answer text. That matters because the generative metrics people usually reach
for (RAGAS faithfulness, answer relevancy) are LLM-as-judge, costing several calls each — and this
system is already rate-limited with *one* call per query. Judged metrics therefore belong in an
offline dataset run, never in the request path. These belong on every query.

**The call graph is an oracle.** For a dependency or change-impact question the correct answer is
known exactly: `get_callers(symbol)` returns every call site in the repository. So `graph_recall`
needs no judge and cannot be wrong about what the right answer was — it can only be wrong about
whether the answer *said* it, which is the caveat below.

**The caveat, stated plainly.** Scoring an answer means matching symbol names against prose. A
model writing `` `process_checkout()` `` with parentheses, hyphenating across a line break, or
splitting a name across table cells can be missed. These are trend signals, not measurements. A
falling `graph_recall` is worth investigating; a single 0.83 is not worth interpreting.
"""
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("observability.scorers")

# Intents for which the graph can adjudicate the answer. For CODE_LOCATION or
# FEATURE_EXPLANATION the correct answer is not a caller list, so recall against one is meaningless.
GRAPH_ADJUDICATED_INTENTS = ("DEPENDENCY_ANALYSIS", "CHANGE_IMPACT")


def _mentions(answer: str, symbol: str) -> bool:
    """
    Whether an answer names a symbol.

    Matches the dotted name and its tail, so `DiscountService.calculate_discount` is found when the
    answer says `calculate_discount`. Word-boundary anchored, because a substring test would count
    `calculate_discount` as present in `test_calculate_discount_rounding` — the same class of bug
    that made the intent classifier read "unchanged" as "change".
    """
    if not symbol or not answer:
        return False
    candidates = {symbol, symbol.rsplit(".", 1)[-1]}
    for candidate in candidates:
        if not candidate:
            continue
        if re.search(rf"(?<![\w.]){re.escape(candidate)}(?![\w])", answer):
            return True
    return False


def _cited_paths(answer: str) -> List[str]:
    """File paths the answer mentions, from backticks, tables or bare prose."""
    pattern = r"[\w./-]+\.(?:py|js|jsx|ts|tsx|java|sql|go|rb|php|json|yaml|yml|md)"
    return list(dict.fromkeys(re.findall(pattern, answer or "")))


def citation_validity(sources: List[Dict[str, Any]], all_chunks: List[Dict[str, Any]]) -> Optional[float]:
    """
    Proportion of returned citations that resolve to a real file in the index.

    A citation the viewer cannot open is worse than no citation: it looks authoritative and leads
    nowhere. This is the machine-readable form of the same check `/snippet` performs on click.
    """
    if not sources:
        return None
    known = {c.get("file_path") for c in all_chunks}
    resolved = sum(1 for s in sources if s.get("file_path") in known)
    return round(resolved / len(sources), 4)


def citation_line_validity(sources: List[Dict[str, Any]], all_chunks: List[Dict[str, Any]]) -> Optional[float]:
    """
    Proportion of citations whose line range actually exists in the cited file.

    Stricter than `citation_validity`: the file may exist while the range points past its end,
    which is what a stale citation after a sync looks like.
    """
    if not sources:
        return None
    extents: Dict[str, int] = {}
    for chunk in all_chunks:
        path = chunk.get("file_path")
        if path:
            extents[path] = max(extents.get(path, 0), int(chunk.get("end_line") or 0))

    valid = 0
    for source in sources:
        path = source.get("file_path")
        start = int(source.get("start_line") or 0)
        if path in extents and 0 < start <= extents[path]:
            valid += 1
    return round(valid / len(sources), 4)


def graph_recall(answer: str, graph_callers: List[Dict[str, Any]]) -> Optional[float]:
    """
    Of the callers the graph knows about, how many did the answer name?

    This is the `5/13 → 13/13` measurement, running on every query instead of once by hand. Returns
    None when the graph knows of no callers — a score of 1.0 for "there was nothing to find" would
    quietly inflate the average.
    """
    names = {c.get("caller_symbol") for c in (graph_callers or []) if c.get("caller_symbol")}
    if not names:
        return None
    named = sum(1 for n in names if _mentions(answer, n))
    return round(named / len(names), 4)


def graph_precision(answer: str, graph_callers: List[Dict[str, Any]],
                    all_chunks: List[Dict[str, Any]],
                    target_symbol: str = "") -> Optional[float]:
    """
    Of the symbols the answer presents as callers, how many are real callers?

    Catches the opposite failure from recall: an answer that invents a plausible caller. Scoped to
    symbols the repository actually defines, so ordinary prose is not counted as a false claim.

    **The subject is excluded.** An answer to "what calls `calculate_discount`?" names
    `calculate_discount` throughout — it is the thing being asked about, not a claimed caller of
    itself. Counting it scored a perfectly correct answer at 0.67, which is the sort of quiet
    unfairness that makes a dashboard untrustworthy.
    """
    truth = {c.get("caller_symbol") for c in (graph_callers or []) if c.get("caller_symbol")}
    if not truth:
        return None

    tails = {t.rsplit(".", 1)[-1] for t in truth}
    excluded = set()
    if target_symbol:
        excluded = {target_symbol, target_symbol.rsplit(".", 1)[-1]}

    repo_symbols = {c.get("symbol") for c in all_chunks if c.get("symbol")}
    claimed = {
        s for s in repo_symbols
        if _mentions(answer, s)
        and s not in excluded and s.rsplit(".", 1)[-1] not in excluded
    }
    if not claimed:
        return None

    correct = sum(1 for s in claimed if s in truth or s.rsplit(".", 1)[-1] in tails)
    return round(correct / len(claimed), 4)


def path_grounding(answer: str, sources: List[Dict[str, Any]],
                   graph_callers: Optional[List[Dict[str, Any]]] = None) -> Optional[float]:
    """
    Proportion of file paths in the answer that the model was actually shown.

    A cheap proxy for faithfulness: a path the model never saw is a path it invented. Not a
    substitute for RAGAS faithfulness, which reasons about claims rather than strings.

    **The graph facts count as "shown", not just the retrieved snippets.** The prompt puts two
    things in front of the model, and rule A1 instructs it to report the complete caller list from
    the facts even when the snippets show fewer. Scoring against snippets alone therefore punished
    the model for obeying the prompt: a verified-correct answer naming all thirteen callers, of
    which retrieval returned six, scored 0.38 and read as heavy hallucination. A correctness
    metric that fires on correct behaviour is worse than no metric — it sends you looking for a
    bug that is not there, or worse, "fixing" the prompt until the answer gets wronger.
    """
    mentioned = _cited_paths(answer)
    if not mentioned:
        return None

    provided = {s.get("file_path") for s in (sources or [])}
    provided |= {c.get("file_path") for c in (graph_callers or [])}
    provided = {p for p in provided if p}

    grounded = sum(1 for p in mentioned if any(p in q or q in p for q in provided))
    return round(grounded / len(mentioned), 4)


def retrieval_hit(target_symbol: str, sources: List[Dict[str, Any]]) -> Optional[bool]:
    """
    Was the symbol the question was about actually among the retrieved chunks?

    False means the answer was produced without the subject's own source in front of the model —
    survivable when the graph facts carry the answer, worth knowing either way.
    """
    if not target_symbol:
        return None
    symbols = {s.get("symbol") for s in (sources or [])}
    tail = target_symbol.rsplit(".", 1)[-1]
    return any(s == target_symbol or (s or "").rsplit(".", 1)[-1] == tail for s in symbols)


def prompt_budget_used(prompt_chars: int, budget_chars: int) -> Optional[float]:
    """
    Fraction of the prompt budget consumed. Trends toward 1.0 are the early warning that
    rate-limit fallbacks are coming back.
    """
    if not budget_chars:
        return None
    return round(min(prompt_chars / budget_chars, 2.0), 4)


def collect(
    *,
    answer: str,
    sources: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    intent: str,
    target_symbol: str = "",
    graph_callers: Optional[List[Dict[str, Any]]] = None,
    prompt_chars: int = 0,
    budget_chars: int = 0,
    degraded: bool = False,
    facts_truncated: bool = False,
) -> Dict[str, Any]:
    """
    Computes every applicable score, omitting those that do not apply.

    A metric that cannot be computed is left out rather than defaulted, so an average over the
    dashboard is an average over queries the metric meant something for.
    """
    # Coerced rather than trusted. These arrive from a response payload that has passed through
    # an LLM call, a fallback formatter and a serialisation boundary; a scorer should not be the
    # thing that discovers one of them was None.
    answer = answer or ""
    sources = sources or []
    all_chunks = all_chunks or []
    graph_callers = graph_callers or []

    scores: Dict[str, Any] = {}

    def put(name: str, value) -> None:
        if value is not None:
            scores[name] = value

    put("citation_validity", citation_validity(sources, all_chunks))
    put("citation_line_validity", citation_line_validity(sources, all_chunks))
    put("path_grounding", path_grounding(answer, sources, graph_callers))
    put("retrieval_hit", retrieval_hit(target_symbol, sources))
    put("prompt_budget_used", prompt_budget_used(prompt_chars, budget_chars))
    scores["degraded"] = bool(degraded)
    scores["facts_truncated"] = bool(facts_truncated)

    # Only scored where the graph can adjudicate. Elsewhere the correct answer is not a caller
    # list, and recall against one would be measuring the wrong thing.
    if intent in GRAPH_ADJUDICATED_INTENTS and graph_callers:
        put("graph_recall", graph_recall(answer, graph_callers))
        put("graph_precision", graph_precision(answer, graph_callers, all_chunks, target_symbol))

    return scores
