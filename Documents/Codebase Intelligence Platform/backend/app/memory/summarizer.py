import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable

from app.observability import prompts
from app.observability.prompts import SUMMARY_PROMPT

logger = logging.getLogger("memory.summarizer")


class ConversationSummarizer:
    """
    Maintains a rolling summary of a conversation so long sessions stay coherent once the
    replayed message window (CodebaseAgentOrchestrator.MAX_HISTORY_MESSAGES) has scrolled past
    the earliest turns.

    The previous implementation concatenated literal strings — "User inquired: '...'" — and kept
    the last four. That is a transcript excerpt, not a summary: it grew without bound in detail
    while losing the actual thread of the investigation.

    The LLM is asked to summarize only once a conversation is long enough to need it; short
    conversations are fully replayed as messages anyway, so summarizing them would spend a
    request for nothing. If no LLM is configured or the call fails, the deterministic fallback
    keeps the previous behaviour rather than losing the summary entirely.
    """

    # Below this much conversation, the raw message window already covers everything.
    SUMMARY_TRIGGER_CHARS = 3000
    MAX_SUMMARY_CHARS = 900
    MAX_TRANSCRIPT_CHARS = 6000

    # Re-summarising on every turn meant two upstream calls per message — one for the answer,
    # one for the summary — which doubled token consumption and, on a rate-limited tier, made
    # the answer itself more likely to be throttled. A summary does not change materially
    # between adjacent turns, so it is refreshed periodically and the cheap deterministic
    # update carries the turns in between.
    SUMMARY_EVERY_N_TURNS = 3

    # The text itself lives in `observability/prompts.py`, where it can be versioned in Langfuse
    # with this literal as the fallback. Kept as a class attribute so existing callers and tests
    # that reference `SYSTEM_PROMPT` are unaffected.
    SYSTEM_PROMPT = SUMMARY_PROMPT

    @classmethod
    def _transcript(cls, messages: List[Dict[str, Any]]) -> str:
        """Renders stored messages as a plain transcript for summarization."""
        lines = []
        for message in messages:
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (message.get("content") or "").strip()
            if content:
                lines.append(f"{role.upper()}: {content}")

        transcript = "\n\n".join(lines)
        if len(transcript) > cls.MAX_TRANSCRIPT_CHARS:
            # Keep the most recent material; the older part is already folded into the
            # existing summary that is passed alongside.
            transcript = "… (earlier turns omitted)\n\n" + transcript[-cls.MAX_TRANSCRIPT_CHARS:]
        return transcript

    @classmethod
    def _conversation_size(cls, messages: List[Dict[str, Any]]) -> int:
        return sum(len(m.get("content") or "") for m in messages)

    @classmethod
    def should_summarize(cls, messages: List[Dict[str, Any]]) -> bool:
        """
        True when the conversation is long enough to need a summary *and* this turn is one of
        the periodic refresh points.
        """
        if cls._conversation_size(messages) < cls.SUMMARY_TRIGGER_CHARS:
            return False

        turns = len(messages) // 2
        return turns % cls.SUMMARY_EVERY_N_TURNS == 0

    @staticmethod
    def deterministic_summary(
        existing_summary: str,
        user_msg: str,
        working_ctx: Dict[str, Any]
    ) -> str:
        """
        Offline fallback used when no LLM is configured or the summarization call fails.
        Deliberately terse — it records the thread of the investigation rather than quoting it.
        """
        feature = (working_ctx or {}).get("current_feature", "")
        symbol = (working_ctx or {}).get("current_symbol", "")

        parts = []
        if existing_summary:
            parts.append(existing_summary)

        focus = []
        if feature:
            focus.append(f"feature '{feature}'")
        if symbol:
            focus.append(f"symbol '{symbol}'")

        latest = user_msg.strip().replace("\n", " ")
        if len(latest) > 90:
            latest = latest[:90] + "…"

        if focus:
            parts.append(f"Investigating {' and '.join(focus)}; last asked: \"{latest}\".")
        else:
            parts.append(f'Last asked: "{latest}".')

        summary = " ".join(parts)
        if len(summary) > ConversationSummarizer.MAX_SUMMARY_CHARS:
            summary = "…" + summary[-ConversationSummarizer.MAX_SUMMARY_CHARS:]
        return summary

    @classmethod
    async def update(
        cls,
        existing_summary: str,
        messages: List[Dict[str, Any]],
        user_msg: str,
        assistant_msg: str,
        working_ctx: Dict[str, Any],
        llm_call: Optional[Callable[..., Awaitable[str]]] = None,
    ) -> str:
        """
        Returns the updated conversation summary.

        `llm_call` is injected rather than imported so this module stays free of a dependency
        on the agent package (which imports memory, and would otherwise cycle).
        """
        full_history = list(messages or []) + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]

        if llm_call and cls.should_summarize(full_history):
            try:
                prompt = (
                    f"Existing summary (may be empty):\n{existing_summary or '(none)'}\n\n"
                    f"Conversation transcript:\n{cls._transcript(full_history)}\n\n"
                    "Write the updated summary."
                )
                summary_prompt = prompts.get(prompts.SUMMARY)
                result = await llm_call(summary_prompt.text, prompt)
                result = (result or "").strip()

                if result:
                    if len(result) > cls.MAX_SUMMARY_CHARS:
                        result = result[: cls.MAX_SUMMARY_CHARS].rstrip() + "…"
                    return result

                logger.warning("Summarizer returned empty output; using deterministic fallback.")
            except Exception as e:
                logger.error("Summarization failed (%s); using deterministic fallback.", e)

        return cls.deterministic_summary(existing_summary, user_msg, working_ctx)
