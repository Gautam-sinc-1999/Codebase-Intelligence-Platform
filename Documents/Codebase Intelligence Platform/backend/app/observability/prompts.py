"""
Prompts, versioned in Langfuse with the in-code text as the fallback.

A prompt is the part of this system most likely to be changed and least likely to be measured.
Rule A1 in the answering prompt — "the facts are authoritative, never describe them as partial" —
exists because of an observed failure, and if someone rewords it and faithfulness drops, nothing
in the current arrangement records which wording produced which number. Versioning the prompt and
linking it to the generation is what makes that attributable.

**The literals below remain the source of truth for behaviour.** Langfuse holds copies that can be
edited and labelled; when it is unreachable, unconfigured, or has never been seeded, the text here
is used instead and the product behaves exactly as it did before any of this existed. That is the
same rule the rest of the integration follows: an observability service being down degrades
observability, nothing else.

Deliberately *not* managed here: the judge prompts in `ragas_metrics.py`. A measuring instrument
that can be changed without a deploy is one that makes every comparison across runs suspect — a
"regression" might only be the judge being reworded. Those stay pinned in code.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger("observability.prompts")


# ------------------------------------------------------------------ the prompts themselves

ANSWER_PROMPT = (
    "You are a helpful, senior software engineer pair-programming with a developer onboarding to a codebase.\n"
    "GROUNDING RULES (these override style):\n"
    "A. Answer ONLY from the material in the user turn: the Static Analysis Facts and the "
    "Retrieved Code Context. Both come from the repository the developer is asking about — "
    "the facts from its parsed call graph, the context verbatim from its source.\n"
    "A1. The Static Analysis Facts are AUTHORITATIVE and COMPLETE for what they state. The "
    "snippets are the top matches for this question, not the whole repository, so they will "
    "often show fewer call sites than the facts list. When the two appear to disagree, "
    "follow the facts. Never describe a list given in the facts as partial, as 'examples', "
    "or as 'a handful' — report it as the complete set it is, and give the count.\n"
    "B. Never invent file paths, symbol names, line numbers, or behaviour that is not visible in that context. "
    "If the context does not contain the answer, say so plainly and ask which file or symbol to look at.\n"
    "C. When you cite a location, use the exact path and line range shown in the context header.\n"
    "STYLE:\n"
    "1. Answer in natural, friendly conversational language first. Do NOT dump huge markdown templates or full code blocks right away unless explicitly asked.\n"
    "2. Provide a clear, concise summary of how the code or feature works (2-4 sentences).\n"
    "3. Mention key files and function names naturally (e.g. 'Checkout starts in `Checkout.jsx` and calls `checkout_controller.py`').\n"
    "4. At the end, offer 2 relevant follow-up options if the user wants deeper details (e.g., line numbers, full execution flow, or change impact analysis).\n"
    "5. If the user is asking a follow-up question, give specific details directly answering their question."
)

SUMMARY_PROMPT = (
    "You maintain a running summary of a technical conversation between a developer and a "
    "codebase assistant.\n"
    "Write 3-5 sentences in plain prose capturing: which feature or area the developer is "
    "investigating, the specific files and symbols established so far, any conclusions "
    "reached, and what they were asking most recently.\n"
    "Preserve exact file paths and symbol names — they are the thread of the investigation. "
    "Do not invent anything that is not in the transcript. Output only the summary."
)

ANSWER = "codebase-answer"
SUMMARY = "conversation-summary"

REGISTRY: Dict[str, Dict[str, Any]] = {
    ANSWER: {
        "text": ANSWER_PROMPT,
        "labels": ["production"],
        "commit_message": "Grounding rules A-C with A1 asserting graph-fact authority",
    },
    SUMMARY: {
        "text": SUMMARY_PROMPT,
        "labels": ["production"],
        "commit_message": "Running conversation summary, preserving paths and symbols",
    },
}


# ------------------------------------------------------------------ resolution

@dataclass
class ResolvedPrompt:
    """
    A prompt ready to send, plus where it came from.

    `client` is the Langfuse prompt object, or None. It is passed to the generation so the trace
    records which version produced the answer — without that link, versioning gives you rollback
    but not attribution, and attribution was the point.
    """
    name: str
    text: str
    version: Optional[int] = None
    source: str = "fallback"
    client: Any = None
    labels: list = field(default_factory=list)

    @property
    def is_managed(self) -> bool:
        return self.source == "langfuse"

    def describe(self) -> Dict[str, Any]:
        return {"prompt_name": self.name, "prompt_version": self.version,
                "prompt_source": self.source}


def prompts_enabled() -> bool:
    return str(settings.LANGFUSE_PROMPTS_ENABLED).strip().lower() not in ("0", "false", "no", "")


def get(name: str, **variables) -> ResolvedPrompt:
    """
    Returns the prompt to use for `name`, preferring the version labelled `production`.

    Never raises and never returns empty: every failure path falls back to the in-code text. A
    prompt is not an optional part of a request — if this returned nothing, the answer would be
    ungrounded rather than merely untraced.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"No prompt registered under '{name}'")

    fallback_text = spec["text"]
    if variables:
        fallback_text = _fill(fallback_text, variables)

    if not prompts_enabled():
        return ResolvedPrompt(name=name, text=fallback_text, source="disabled")

    from app.observability import tracing

    client = tracing._resolve_client()
    if client is None:
        return ResolvedPrompt(name=name, text=fallback_text)

    try:
        managed = client.get_prompt(
            name,
            cache_ttl_seconds=settings.LANGFUSE_PROMPT_CACHE_TTL,
            # Handed to the SDK as well as used below: this is what lets it serve something
            # sensible on a cold cache when Langfuse cannot be reached at all.
            fallback=spec["text"],
        )
    except Exception as e:
        logger.warning("Could not fetch prompt '%s' (%s); using the in-code text.", name, e)
        return ResolvedPrompt(name=name, text=fallback_text)

    if getattr(managed, "is_fallback", False):
        # The SDK served our own text back because it could not reach Langfuse. Reporting that
        # as a managed version would attribute a run to a version that was never used.
        return ResolvedPrompt(name=name, text=fallback_text)

    try:
        text = managed.compile(**variables) if variables else managed.compile()
    except Exception as e:
        logger.warning("Prompt '%s' v%s did not compile (%s); using the in-code text.",
                       name, getattr(managed, "version", "?"), e)
        return ResolvedPrompt(name=name, text=fallback_text)

    if not isinstance(text, str) or not text.strip():
        logger.warning("Prompt '%s' compiled to nothing; using the in-code text.", name)
        return ResolvedPrompt(name=name, text=fallback_text)

    return ResolvedPrompt(
        name=name,
        text=text,
        version=getattr(managed, "version", None),
        source="langfuse",
        client=managed,
        labels=list(getattr(managed, "labels", []) or []),
    )


def _fill(text: str, variables: Dict[str, Any]) -> str:
    """Mustache substitution for the fallback path, matching what Langfuse does server-side."""
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
