"""
The evaluation judge — a second LLM, deliberately not the one that answers questions.

RAGAS metrics are almost all LLM-as-judge: faithfulness decomposes an answer into claims and checks
each against the context, so six metrics over one item can mean 15-30 calls. The product already
makes one call per query and is rate-limited, so a judge sharing its provider would make evaluation
and answering compete for the same budget — and the loser would be the user waiting for an answer.

Configured through `JUDGE_*` rather than `LLM_*` for exactly that reason. It speaks the OpenAI
chat-completions schema, as the product's providers do, so pointing it at Gemini, Groq, Cerebras or
a local Ollama is four environment variables and no code.

Everything here is guarded and returns `None` rather than raising: an evaluation harness that dies
halfway through leaves you with neither the metric nor the run.
"""
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger("observability.judge")

# Worth retrying: the request was fine, the service momentarily was not. Free tiers return 429
# frequently by design, so this is the normal path rather than an exceptional one.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


def judge_configured() -> bool:
    return bool(settings.JUDGE_API_KEY)


def describe_judge() -> Dict[str, Any]:
    """What the harness reports it measured with. A score is meaningless without its judge."""
    return {
        "provider": settings.JUDGE_PROVIDER,
        "model": settings.JUDGE_MODEL,
        "configured": judge_configured(),
    }


class Judge:
    """
    Issues judging calls, paced so a free tier does not reject the run.

    Concurrency is capped rather than unbounded because free tiers limit requests *per minute* far
    more tightly than per day: firing a dataset's worth of judgements at once fails the whole run,
    while pacing them merely makes it slower.
    """

    def __init__(self, *, concurrency: Optional[int] = None, model: Optional[str] = None):
        self.model = model or settings.JUDGE_MODEL
        self._semaphore = asyncio.Semaphore(concurrency or settings.JUDGE_MAX_CONCURRENCY)
        self.calls = 0

    @property
    def url(self) -> str:
        return f"{settings.JUDGE_BASE_URL.rstrip('/')}/chat/completions"

    async def ask(self, client: httpx.AsyncClient, prompt: str, *,
                  system: str = "You are a strict evaluator. Reply with JSON only.",
                  temperature: float = 0.0) -> Optional[str]:
        """
        One judging call. Returns the reply text, or None if it could not be obtained.

        Temperature is zero because a metric that changes between runs of the same data cannot be
        used to detect a regression — which is the only reason to compute it.
        """
        if not judge_configured():
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {settings.JUDGE_API_KEY}"}

        async with self._semaphore:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        self.url, json=payload, headers=headers,
                        timeout=settings.JUDGE_TIMEOUT_SECONDS,
                    )
                except Exception as e:
                    logger.warning("Judge call failed (%s); attempt %d/%d", e, attempt, MAX_ATTEMPTS)
                    if attempt == MAX_ATTEMPTS:
                        return None
                    await asyncio.sleep(min(2 ** attempt, 15))
                    continue

                if response.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                    # Prefer what the provider actually said to wait; a 429 is a timing failure,
                    # not a bad request.
                    delay = _retry_after(response) or min(2 ** attempt, 15)
                    logger.info("Judge returned %d; retrying in %.1fs", response.status_code, delay)
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 400:
                    logger.warning("Judge returned %d: %s",
                                   response.status_code, response.text[:200])
                    return None

                self.calls += 1
                try:
                    return response.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    logger.warning("Judge reply was not in the expected shape: %s", e)
                    return None
        return None

    async def ask_json(self, client: httpx.AsyncClient, prompt: str, **kwargs) -> Optional[Any]:
        """A judging call whose reply is parsed as JSON, tolerating the usual model noise."""
        raw = await self.ask(client, prompt, **kwargs)
        return parse_json(raw)


def _retry_after(response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return min(float(value), 30.0)
    except (TypeError, ValueError):
        return None


def parse_json(raw: Optional[str]) -> Optional[Any]:
    """
    Pulls JSON out of a model reply.

    Models wrap JSON in prose and fenced code blocks however firmly they are told not to, and a
    metric that returns None because of a stray ```json fence is a metric that silently stops
    measuring. Tried in order: the whole string, a fenced block, then the first balanced object
    or array.
    """
    if not raw:
        return None

    candidates: List[str] = [raw.strip()]

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.S)
    if fenced:
        candidates.append(fenced.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            candidates.append(raw[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue

    logger.debug("Could not parse a judge reply as JSON: %.120s", raw)
    return None
