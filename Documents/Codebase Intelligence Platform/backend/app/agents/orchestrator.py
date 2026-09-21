import logging
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator

import httpx

from app.core.config import settings
from app.retrieval.intent_classifier import QueryIntentClassifier
from app.retrieval.hybrid_retriever import HybridRetriever
from app.graph.feature_tracer import FeatureTracer
from app.agents.impact_analyzer import ChangeImpactAnalyzer
from app.graph.neo4j_client import neo4j_client
from app.memory.conversation_memory import ConversationMemory
from app.memory.summarizer import ConversationSummarizer

logger = logging.getLogger("agent.orchestrator")

# One pooled AsyncClient for the process. Creating a client per request would discard connection
# reuse and TLS session caching, which matters when every chat turn makes an upstream call.
_http_client: Optional[httpx.AsyncClient] = None
_http_client_lock = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    """Returns the shared HTTP client, creating it on first use."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        async with _http_client_lock:
            if _http_client is None or _http_client.is_closed:
                total = CodebaseAgentOrchestrator.LLM_TIMEOUT_SECONDS
                _http_client = httpx.AsyncClient(
                    # Connect must never exceed the overall budget, or a host that accepts no
                    # connection would stall past the timeout the caller actually configured.
                    timeout=httpx.Timeout(total, connect=min(10.0, total)),
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                )
    return _http_client


async def close_http_client() -> None:
    """Closes the shared HTTP client. Called on application shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None

class CodebaseAgentOrchestrator:
    """
    Query orchestration pipeline: intent classification -> hybrid retrieval -> graph analysis ->
    LLM reasoning (Groq / OpenAI), with rule-based formatters as the fallback.

    Named for what it is. This was previously called a "LangGraph-inspired Agentic Execution
    Graph" in a module named langgraph_agent.py, while being a straight-line function with no
    graph, no nodes, no branching state machine and no LangGraph dependency installed. The name
    described an architecture the code did not have, in the file most central to the product.

    If branching per intent, tool loops or retries are wanted later, adopting LangGraph for real
    is a reasonable step — but the name should follow the code rather than lead it.

    Response Philosophy:
    - Conversational & Progressive: Starts with concise, high-level natural language answers.
    - Contextual Memory: Keeps track of current feature & active symbols across turns.
    - Progressive Detail: Expands into full line-by-line snippets or call graphs on follow-ups.
    """

    LLM_TIMEOUT_SECONDS = 30.0

    # How much prior conversation to replay to the model. Six messages is three exchanges —
    # enough for "show me the line numbers" to resolve against what was just discussed, without
    # spending the context budget that F-04's retrieved code now occupies.
    MAX_HISTORY_MESSAGES = 6
    MAX_HISTORY_CHARS_PER_MESSAGE = 600

    @classmethod
    def _build_messages(
        cls,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """
        Assembles the chat messages array: system prompt, recent turns, then the current turn.

        Prior turns were previously dropped entirely — only a crude pseudo-summary was passed —
        so every follow-up arrived with no idea what "it" or "that function" referred to, even
        though the prompt explicitly invites follow-ups.
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Filtered before slicing, not after. A thread also carries system markers (a repository
        # sync records a version boundary), and slicing first would let those consume history
        # slots and push real turns out of the window they exist to preserve.
        conversational = [
            m for m in (history or [])
            if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
        ]

        # The history share is enforced here, against the real messages. Subtracting an assumed
        # share in `_build_context` while letting the actual history be twice that put the prompt
        # 30 % over budget at a tight setting — a budget computed from assumptions is not a budget.
        history_budget = int(cls._char_budget() * cls.HISTORY_BUDGET_SHARE)
        spent_on_history = 0
        kept: List[Dict[str, str]] = []

        # Walked newest-first so that when the budget runs out it is the *oldest* turns that are
        # dropped. Iterating oldest-first and skipping once full does the opposite — it discards
        # the most recent exchange, which is precisely the one a follow-up like "who calls it?"
        # refers back to.
        for message in reversed(conversational[-cls.MAX_HISTORY_MESSAGES:]):
            role = message.get("role")
            content = (message.get("content") or "").strip()

            # Stored assistant answers can be long; the gist is enough to resolve a reference.
            if len(content) > cls.MAX_HISTORY_CHARS_PER_MESSAGE:
                content = content[: cls.MAX_HISTORY_CHARS_PER_MESSAGE] + "… (truncated)"

            if spent_on_history + len(content) > history_budget:
                break
            spent_on_history += len(content)

            kept.append({"role": role, "content": content})

        # Restored to chronological order: the model reads a conversation, not a stack.
        messages.extend(reversed(kept))

        messages.append({"role": "user", "content": user_prompt})
        return messages

    @classmethod
    def _resolve_provider(cls) -> Optional[Dict[str, Any]]:
        """
        Picks the configured LLM provider. Both speak the OpenAI chat-completions schema, so
        only the endpoint, key and model differ — no need for two near-identical call paths.
        """
        provider = settings.LLM_PROVIDER.lower().strip()

        if settings.GROQ_API_KEY and provider in ("groq", "auto"):
            return {
                "name": "Groq",
                "url": f"{settings.GROQ_BASE_URL.rstrip('/')}/chat/completions",
                "key": settings.GROQ_API_KEY,
                "model": settings.GROQ_MODEL,
            }

        if settings.OPENAI_API_KEY and provider in ("openai", "auto"):
            return {
                "name": "OpenAI",
                "url": "https://api.openai.com/v1/chat/completions",
                "key": settings.OPENAI_API_KEY,
                "model": settings.DEFAULT_LLM_MODEL,
            }

        return None

    # Statuses worth trying again: the request was fine, the service was momentarily not.
    RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
    LLM_MAX_ATTEMPTS = 3
    MAX_RETRY_DELAY_SECONDS = 10.0

    @classmethod
    def _retry_delay(cls, response, attempt: int) -> float:
        """
        How long to wait before retrying, preferring what the provider actually told us.

        A 429 is a *timing* failure, not a bad request: Groq replies with `Retry-After` and
        states the wait in the body ("Please try again in 4.1s"). Discarding that and serving a
        rule-based answer instead makes a two-second hiccup look like a permanently degraded
        product.
        """
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(float(header), cls.MAX_RETRY_DELAY_SECONDS)
            except ValueError:
                pass

        try:
            match = re.search(r"try again in ([0-9.]+)s", response.text)
            if match:
                return min(float(match.group(1)) + 0.25, cls.MAX_RETRY_DELAY_SECONDS)
        except Exception:
            pass

        return min(2.0 ** attempt, cls.MAX_RETRY_DELAY_SECONDS)

    @classmethod
    async def call_llm_provider(
        cls,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Executes LLM reasoning against Groq or OpenAI over a shared async HTTP client.

        Uses httpx rather than urllib.request: the previous implementation called a blocking
        urlopen() from inside this coroutine, which stalled the whole event loop for the full
        timeout. That made the server effectively single-concurrency — one chat at a time, with
        every other request queued behind it.

        Returns "" on any failure so the caller falls back to the rule-based formatters.
        """
        provider = cls._resolve_provider()
        if not provider:
            return ""

        logger.info("Requesting completion from %s (model: %s)", provider["name"], provider["model"])

        payload = {
            "model": provider["model"],
            "messages": cls._build_messages(system_prompt, user_prompt, history),
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {provider['key']}",
            "Content-Type": "application/json",
            "User-Agent": "CodebaseIntelligence/1.0",
        }

        for attempt in range(cls.LLM_MAX_ATTEMPTS):
            try:
                client = await get_http_client()
                response = await client.post(provider["url"], json=payload, headers=headers)

                if response.status_code in cls.RETRYABLE_STATUS and attempt < cls.LLM_MAX_ATTEMPTS - 1:
                    delay = cls._retry_delay(response, attempt)
                    logger.warning(
                        "%s returned HTTP %s; retrying in %.1fs (attempt %d/%d)",
                        provider["name"], response.status_code, delay,
                        attempt + 1, cls.LLM_MAX_ATTEMPTS,
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.status_code != 200:
                    # A non-200 previously fell through silently, so an expired key or a rate
                    # limit was indistinguishable from a working system answering from fallbacks.
                    logger.error(
                        "%s returned HTTP %s: %s",
                        provider["name"], response.status_code, response.text[:300],
                    )
                    return ""

                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    logger.error("%s returned no choices: %s", provider["name"], str(body)[:300])
                    return ""

                return choices[0].get("message", {}).get("content", "") or ""

            except httpx.TimeoutException:
                logger.error("%s request timed out after %ss", provider["name"], cls.LLM_TIMEOUT_SECONDS)
                return ""
            except httpx.HTTPError as e:
                # Transport failures are often transient too (connection reset, DNS blip).
                if attempt < cls.LLM_MAX_ATTEMPTS - 1:
                    logger.warning("%s transport error (%s); retrying.", provider["name"], e)
                    await asyncio.sleep(min(2.0 ** attempt, cls.MAX_RETRY_DELAY_SECONDS))
                    continue
                logger.error("%s transport error: %s", provider["name"], e)
                return ""
            except (KeyError, ValueError) as e:
                logger.error("%s returned an unreadable response: %s", provider["name"], e)
                return ""

        return ""

    # Character budgets for the retrieved-code section of the prompt. Characters rather than
    # tokens keeps this dependency-free; ~4 chars/token means this lands near 3k tokens of code.
    MAX_CODE_CHARS_TOTAL = 12000
    MAX_CODE_CHARS_PER_SOURCE = 2500

    # Maps our internal language ids onto markdown fence hints.
    _FENCE_LANG = {
        "python": "python", "javascript": "javascript", "typescript": "typescript",
        "react_jsx": "jsx", "react_tsx": "tsx", "java": "java", "sql": "sql",
        "html": "html", "css": "css", "yaml": "yaml", "json": "json", "markdown": "markdown",
    }

    # Words that look like identifiers but are just English. Without this, "change" in
    # "if I change X" matches any symbol containing "change".
    _QUERY_STOPWORDS = {
        "what", "where", "when", "which", "who", "why", "how", "does", "do", "did", "is", "are",
        "was", "were", "the", "a", "an", "in", "on", "at", "to", "of", "for", "from", "and", "or",
        "if", "i", "it", "its", "this", "that", "there", "here", "me", "my", "we", "you", "your",
        "change", "changes", "modify", "modifies", "affect", "affects", "affected", "break",
        "breaks", "impact", "impacts", "depend", "depends", "depending", "call", "calls", "called",
        "caller", "callers", "by", "used", "uses", "use", "using", "import", "imports", "find",
        "locate", "file", "files", "line", "lines", "work", "works", "working", "explain", "show",
        "tell", "give", "get", "list", "code", "repo", "repository", "function", "functions",
        "class", "classes", "method", "methods", "module", "feature", "logic", "implementation",
        "happen", "happens", "would", "will", "can", "should", "about", "with", "into", "all", "any",
    }

    # Entity kinds that represent real code, preferred over file-level 'module'/'table' stand-ins.
    _CODE_ENTITY_TYPES = {"function", "method", "class", "component", "endpoint"}

    # Words that refer back to something already under discussion rather than naming it.
    _ANAPHORA = {
        "it", "its", "it's", "this", "that", "these", "those", "them", "they",
        "same", "above", "there", "one",
    }

    # Structural vocabulary — words that describe how code is organised rather than what the
    # product does. A repository is full of these and none of them is a "feature".
    _GENERIC_PATH_TOKENS = {
        "api", "app", "apps", "src", "lib", "libs", "backend", "frontend", "server", "client",
        "web", "service", "services", "model", "models", "controller", "controllers", "router",
        "routers", "route", "routes", "component", "components", "util", "utils", "helper",
        "helpers", "core", "common", "shared", "config", "configs", "setting", "settings",
        "test", "tests", "spec", "specs", "main", "index", "init", "base", "type", "types",
        "schema", "schemas", "database", "migration", "migrations", "static", "public",
        "asset", "assets", "style", "styles", "doc", "docs", "script", "scripts", "build",
        "dist", "module", "modules", "internal", "handler", "handlers", "middleware",
        "repository", "repositories", "store", "stores", "view", "views", "page", "pages",
        "hook", "hooks", "context", "provider", "providers", "interface", "impl", "class",
        "function", "method", "value", "values", "data", "item", "items", "list", "dict",
    }

    # Feature vocabulary is derived per repository and reused across turns.
    _feature_vocab_cache: Dict[str, set] = {}

    @staticmethod
    def _split_identifier(name: str) -> List[str]:
        """Splits snake_case, kebab-case, dotted and camelCase names into lowercase words."""
        if not name:
            return []
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
        return [part.lower() for part in re.split(r"[^A-Za-z0-9]+", spaced) if part]

    @classmethod
    def _build_feature_vocabulary(cls, repository_id: str, all_chunks: List[Dict[str, Any]]) -> set:
        """
        Derives the set of words that name features *in this repository*, from its directory
        names, file names and symbol names — with structural terms removed.

        Replaces a hardcoded list of six ecommerce words, which meant feature tracing only ever
        worked for the bundled sample repo; any other domain fell through to no feature at all.
        """
        cache_key = f"{repository_id}:{len(all_chunks)}"
        cached = cls._feature_vocab_cache.get(cache_key)
        if cached is not None:
            return cached

        def admissible(token: str) -> bool:
            return (len(token) >= 4 and not token.isdigit()
                    and token not in cls._GENERIC_PATH_TOKENS)

        # Whole names — a directory, a file stem, or a complete symbol — name something outright.
        whole_names = set()
        # The leading token of a file or directory name. A file is named after what it is for,
        # and the first token is its subject: `payment_service.py` is about payments. This is what
        # keeps a single-file feature admissible without readmitting fragments from the middle of
        # a symbol name.
        leading_tokens = set()
        # Fragments are only pieces of a name, so they are counted by how many distinct *files*
        # they appear in. A theme that runs through a codebase shows up in several; a fragment of
        # one symbol's name shows up in one.
        fragment_files: Dict[str, set] = {}

        for chunk in all_chunks:
            file_path = chunk.get("file_path", "")
            symbol = chunk.get("symbol", "")

            segments = [seg for seg in re.split(r"[/\\]", file_path) if seg]
            for index, segment in enumerate(segments):
                is_file = index == len(segments) - 1
                name = re.sub(r"\.[A-Za-z0-9]+$", "", segment) if is_file else segment
                if admissible(name.lower()):
                    whole_names.add(name.lower())
                name_tokens = cls._split_identifier(name)
                if name_tokens and admissible(name_tokens[0]):
                    leading_tokens.add(name_tokens[0])
                for token in name_tokens:
                    if admissible(token):
                        fragment_files.setdefault(token, set()).add(file_path)

            for candidate in (symbol, symbol.rsplit(".", 1)[-1]):
                if candidate and admissible(candidate.lower()):
                    whole_names.add(candidate.lower())
            for token in cls._split_identifier(symbol):
                if admissible(token):
                    fragment_files.setdefault(token, set()).add(file_path)

        # `scan_change_point` contributed `point`, and asking about "the point of this" then set
        # the tracked feature to "point" — a fragment of one symbol's name rather than anything
        # the repository is organised around. Requiring a fragment to span more than one file
        # separates a genuine theme ("checkout", which appears in a component, a controller and a
        # service) from an accident of one identifier.
        recurring = {t for t, files in fragment_files.items() if len(files) >= 2}

        vocabulary = whole_names | recurring | leading_tokens
        cls._feature_vocab_cache[cache_key] = vocabulary
        return vocabulary

    @classmethod
    def _extract_target_feature(cls, query: str, repository_id: str, all_chunks: List[Dict[str, Any]]) -> str:
        """Finds the most specific word in the query that names something this repository contains."""
        vocabulary = cls._build_feature_vocabulary(repository_id, all_chunks)
        if not vocabulary:
            return ""

        # Whole identifiers from the query are tried alongside their split tokens. Splitting
        # alone turned "scan_change_point" into scan/change/point, so a user naming the symbol
        # outright matched none of them once fragments stopped being admitted on their own.
        whole_tokens = [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)]
        candidates = whole_tokens + cls._split_identifier(query)

        best = ""
        for token in candidates:
            if len(token) < 4 or token in cls._QUERY_STOPWORDS:
                continue
            # Longest match wins: "checkout" is more specific than "check".
            if token in vocabulary and len(token) > len(best):
                best = token
        return best

    @classmethod
    def _is_followup_reference(cls, query: str) -> bool:
        """
        True when the query points back at the current subject instead of naming a new one —
        "who calls it?", "show me its callers", "what about that function?".

        Such a query names no symbol, so symbol extraction returns nothing and retrieval has
        almost nothing to match on. Falling through to the top-ranked chunk then picks an
        arbitrary symbol; the conversation's established subject is the correct referent.
        """
        return bool(set(re.findall(r"[a-z']+", query.lower())) & cls._ANAPHORA)

    @classmethod
    def _extract_target_symbol(cls, query: str, all_chunks: List[Dict[str, Any]]) -> str:
        """
        Finds the symbol the user actually named, by matching identifier-shaped tokens from the
        query against the repository's indexed symbols.

        Previously the target was simply retrieved_chunks[0]["symbol"] — the highest-ranked
        search hit — so asking "what depends on calculate_discount?" would silently analyse the
        enclosing class DiscountService (which ranks higher as a larger chunk) and answer about
        the wrong thing. Retrieval rank answers "what is relevant", not "what did you name".

        Returns "" when the query names nothing recognisable, letting the caller fall back.
        """
        if not all_chunks:
            return ""

        # Tokens inside backticks are an explicit signal and outrank bare words.
        backticked = {t.strip().lower() for t in re.findall(r"`([^`]+)`", query)}

        bare = set()
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", query):
            low = tok.lower()
            if low in cls._QUERY_STOPWORDS or len(tok) < 3:
                continue
            bare.add(low)

        candidates = backticked | bare
        if not candidates:
            return ""

        best_symbol = ""
        best_score = 0

        for chunk in all_chunks:
            symbol = chunk.get("symbol", "")
            if not symbol:
                continue
            sym_low = symbol.lower()
            # "DiscountService.calculate_discount" -> also match on "calculate_discount"
            tail = sym_low.rsplit(".", 1)[-1]

            for cand in candidates:
                if cand == sym_low:
                    score = 100
                elif cand == tail:
                    score = 90
                elif "." in cand and cand.rsplit(".", 1)[-1] == tail:
                    score = 85
                elif len(cand) > 4 and cand in sym_low:
                    # Weight a partial match by how much of the symbol it accounts for, so
                    # "discount" binds to `DiscountService` (8/15) rather than to
                    # `test_loyalty_tier_discount` (8/26), where it is incidental. Below the
                    # floor the token is not a reference to that symbol at all.
                    coverage = len(cand) / max(len(sym_low), 1)
                    if coverage < 0.34:
                        continue
                    score = 40 + coverage * 30
                else:
                    continue

                if cand in backticked:
                    score += 50
                if chunk.get("entity_type") in cls._CODE_ENTITY_TYPES:
                    score += 5
                # Tie-break toward the more specific (longer) matched token.
                score += min(len(cand), 20) * 0.1

                if score > best_score:
                    best_score = score
                    best_symbol = symbol

        return best_symbol

    # File extensions stripped when matching a query token against a path, so "decide",
    # "decide.py" and "src/ui/decide.py" all reach the same file.
    _PATH_EXTENSIONS = (".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".php")

    @classmethod
    def _extract_target_file(cls, query: str, all_chunks: List[Dict[str, Any]]) -> str:
        """
        Finds the file or module the user named, when they named one rather than a symbol.

        Asking about a *file* is an ordinary way to ask — "if I change the decide module, what
        breaks?" — and there was no path to the right answer for it. `_extract_target_symbol`
        matches symbol names only, a module is not a symbol, so the query matched nothing and the
        caller fell back to the top-ranked retrieval hit. In the case that produced this issue
        that was `a_decision` in `tests/test_decide.py`, and the impact report that followed was
        about a test helper rather than the module asked about.

        Returns "" when no file is named, so the symbol path remains in charge of symbol
        questions.
        """
        if not all_chunks:
            return ""

        backticked = {t.strip().lower() for t in re.findall(r"`([^`]+)`", query)}
        bare = set()
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_./\\-]*", query):
            low = tok.lower().strip("./")
            if low in cls._QUERY_STOPWORDS or len(low) < 3:
                continue
            bare.add(low)

        candidates = backticked | bare
        if not candidates:
            return ""

        def stem(path: str) -> str:
            name = path.rsplit("/", 1)[-1].lower()
            for ext in cls._PATH_EXTENSIONS:
                if name.endswith(ext):
                    return name[: -len(ext)]
            return name

        best_file, best_score = "", 0.0
        # Sorted, not set order. Paths were iterated straight out of a set, so two files scoring
        # equally resolved by hash order — which varies per process, making the same question
        # answerable differently on two runs of the same repository.
        for path in sorted({c.get("file_path", "") for c in all_chunks if c.get("file_path")}):
            low = path.lower()
            filename = low.rsplit("/", 1)[-1]
            file_stem = stem(path)
            segments = set(low.split("/"))

            for cand in candidates:
                cand_stem = stem(cand)
                if cand == low:
                    score = 100.0
                elif cand == filename:
                    score = 95.0
                elif cand_stem and cand_stem == file_stem:
                    # "decide" -> decide.py, not test_decide.py, whose stem is "test_decide".
                    #
                    # Requiring an exact stem rather than containment is belt-and-braces: the
                    # bias away from tests and the shorter-stem tie-break below would each
                    # separate those two on their own. It earns its place by being the rule that
                    # states the intent, so the other two stay tie-breaks rather than becoming
                    # load-bearing by accident.
                    score = 85.0
                elif "/" in cand and low.endswith(cand):
                    score = 80.0
                elif cand in segments:
                    score = 70.0
                else:
                    continue

                # Tie-break away from tests. A question about "the decide module" means the
                # implementation; only a tie-break, so an explicitly named test file still wins
                # on its own exact match.
                if "test" in file_stem or "/tests/" in f"/{low}":
                    score -= 4.0

                score += min(len(cand), 20) * 0.1
                # On an otherwise equal score, the file whose name is closest to what was asked
                # for wins: "decide" means decide.py rather than decide_helpers.py.
                score -= min(len(file_stem), 40) * 0.01
                if cand in backticked:
                    score += 50.0

                if score > best_score:
                    best_score, best_file = score, path

        return best_file

    # Characters per token, for budgeting. A real tokeniser would be exact, but it means a
    # dependency and a model-specific vocabulary; 4 is the standard approximation for English
    # prose and code, and the budget below is set low enough that the error does not matter.
    CHARS_PER_TOKEN = 4

    # How the budget is divided, most valuable last so it receives the remainder.
    #
    # Graph facts are authoritative but compact; history only resolves references; **code is
    # what stops the model inventing behaviour**, so it gets whatever is left rather than a
    # fixed slice.
    FACTS_BUDGET_SHARE = 0.25
    HISTORY_BUDGET_SHARE = 0.15

    @classmethod
    def _char_budget(cls) -> int:
        return max(2000, int(settings.MAX_PROMPT_TOKENS) * cls.CHARS_PER_TOKEN)

    @classmethod
    def _fit_to_budget(cls, text: str, limit: int, what: str) -> str:
        """
        Trims a prompt section to a character limit, saying so where it cuts.

        Silent truncation would be worse than the overrun: the model would read a list that
        stops mid-way as a complete one, which is exactly the failure the facts block exists to
        prevent.
        """
        if len(text) <= limit:
            return text
        kept = text[:limit].rsplit("\n", 1)[0]
        logger.info("Trimmed %s from %d to %d characters to fit the prompt budget.",
                    what, len(text), len(kept))
        return kept + f"\n  … ({what} truncated to fit the prompt budget — the totals above are still exact)"

    # How many graph entries are listed before the block is truncated. A widely-used helper can
    # have hundreds of callers, and spending the whole context budget on them would crowd out the
    # source code the model also needs. Truncation is stated in the text rather than silent.
    MAX_GRAPH_FACTS = 40

    @classmethod
    def _build_graph_facts(
        cls,
        intent: str,
        current_symbol: str,
        repository_id: str,
        execution_flow: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        target_file: str = "",
        all_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Renders what static analysis actually established, for inclusion in the prompt (G-01).

        The call graph was being built, queried, and shown in the UI panel — and then the answer
        was generated from retrieved snippets alone. That is not a missing feature so much as a
        contradiction: the panel and the prose could disagree inside a single response, and the
        prose is what gets read.

        It matters most exactly where retrieval is weakest. Retrieval returns the top `k` chunks
        by relevance, so for a helper used in fifty places it returns a handful of call sites and
        the model reports those as the complete set. The graph has all fifty. Measured on a
        13-caller helper: retrieval surfaced 5, and the answer presented them as exhaustive.

        Returns an empty string when there is nothing established, so a question the graph cannot
        speak to adds nothing to the prompt rather than an empty heading.
        """
        blocks: List[str] = []

        def render(title: str, rows: List[str], note: str = "") -> None:
            if not rows:
                return
            shown = rows[: cls.MAX_GRAPH_FACTS]
            omitted = len(rows) - len(shown)
            body = "\n".join(f"  - {r}" for r in shown)
            if omitted > 0:
                body += f"\n  - …and {omitted} more (list truncated, the total above is exact)"
            blocks.append(f"{title} ({len(rows)} total){note}:\n{body}")

        # Module-level facts, when the question named a file rather than a symbol.
        #
        # "What depends on this module?" is not the same question as "what depends on this
        # function", and answering it by picking one symbol out of the file — or worse, the top
        # retrieval hit from somewhere else entirely — answers something nobody asked.
        if target_file and all_chunks:
            defined = [c for c in all_chunks if c.get("file_path") == target_file]
            render(
                f"SYMBOLS DEFINED IN `{target_file}`",
                [f"`{c.get('symbol')}` ({c.get('entity_type', 'symbol')}) "
                 f"lines {c.get('start_line')}-{c.get('end_line')}" for c in defined],
                note=" — this is the whole file",
            )

            external: List[str] = []
            seen = set()
            for chunk in defined:
                try:
                    callers = neo4j_client.get_callers(chunk.get("symbol", ""), repository_id)
                except Exception as e:
                    logger.warning("Graph caller lookup failed for '%s': %s", chunk.get("symbol"), e)
                    continue
                for caller in callers:
                    # Calls from inside the same file are internal structure, not dependents. A
                    # module's dependents are the things outside it that would break.
                    if caller.get("file_path") == target_file:
                        continue
                    key = (caller.get("caller_symbol"), caller.get("file_path"))
                    if key in seen:
                        continue
                    seen.add(key)
                    external.append(
                        f"`{caller.get('caller_symbol')}` in {caller.get('file_path')} "
                        f"→ calls `{chunk.get('symbol')}`"
                    )

            render(
                f"EXTERNAL DEPENDENTS OF `{target_file}` — everything outside it that calls into it",
                external,
                note=" — this list is complete; changing this file affects exactly these",
            )

        # Callers and dependencies, read straight from the graph.
        #
        # Computed here for every intent that has a subject. Previously only CHANGE_IMPACT and
        # FEATURE_EXPLANATION computed anything at all, so a plain "who calls X?" — the question
        # a call graph exists to answer — reached the model with no graph data whatsoever.
        if current_symbol and repository_id:
            try:
                callers = neo4j_client.get_callers(current_symbol, repository_id)
            except Exception as e:
                logger.warning("Graph caller lookup failed for '%s': %s", current_symbol, e)
                callers = []

            render(
                f"CALLERS OF `{current_symbol}` — every call site in the repository",
                [f"`{c.get('caller_symbol')}` in {c.get('file_path')}"
                 f"{':' + str(c['start_line']) if c.get('start_line') else ''}"
                 for c in callers],
                note=" — this list is complete; do not describe it as partial or as examples",
            )

            if intent in ("DEPENDENCY_ANALYSIS", "ARCHITECTURE"):
                try:
                    deps = neo4j_client.get_forward_dependencies(current_symbol, 2, repository_id)
                except Exception as e:
                    logger.warning("Graph dependency lookup failed for '%s': %s", current_symbol, e)
                    deps = []
                render(
                    f"WHAT `{current_symbol}` DEPENDS ON (up to 2 hops)",
                    [f"`{d.get('symbol')}` ({d.get('label', 'symbol')}) in {d.get('file_path')}"
                     f" — depth {d.get('depth', 1)}" for d in deps],
                )

        # Change impact, already resolved against the graph by ChangeImpactAnalyzer.
        if impact_analysis:
            target = impact_analysis.get("primary_target") or {}
            if target:
                blocks.append(
                    f"CHANGE TARGET: `{target.get('symbol', current_symbol)}` in "
                    f"{target.get('file_path', 'unknown file')}"
                )
            render("CONFIRMED CALLERS (would be affected by a change)",
                   [f"`{c.get('symbol')}` in {c.get('file_path')}"
                    for c in impact_analysis.get("confirmed_callers", [])])
            render("AFFECTED TESTS",
                   [f"`{t.get('symbol')}` in {t.get('file_path')}" if isinstance(t, dict) else str(t)
                    for t in impact_analysis.get("affected_tests", [])])
            render("FILES REQUIRING REVIEW",
                   [str(f) for f in impact_analysis.get("confirmed_files", [])])

        # Execution flow, already traced across layers by FeatureTracer.
        if execution_flow and execution_flow.get("flow_steps"):
            render(
                f"TRACED EXECUTION FLOW for `{execution_flow.get('feature', '')}`",
                [f"step {s.get('step')} — {s.get('layer')}: {s.get('title')} "
                 f"({s.get('file_path')}:{s.get('lines')})"
                 for s in execution_flow["flow_steps"]],
            )

        if not blocks:
            return ""

        return (
            "\n\nSTATIC ANALYSIS FACTS (authoritative — derived from the parsed call graph, "
            "not from the snippets below):\n"
            + "\n\n".join(blocks)
            + "\n\nThese facts were computed by walking the repository's syntax trees and call "
              "graph. They are complete for what they state. Where a snippet below appears to "
              "disagree, or where the snippets show fewer call sites than listed here, the facts "
              "above are correct — the snippets are an excerpt, not the whole repository."
        )

    @classmethod
    def _build_context(
        cls,
        query: str,
        current_feature: str,
        existing_summary: str,
        sources: List[Dict[str, Any]],
        graph_facts: str = "",
        reserved_chars: int = 0
    ) -> str:
        """
        Builds the user-turn context. Critically this includes the actual source code of each
        retrieved chunk — without it the model is asked to explain code it has never seen, which
        is the dominant cause of hallucinated answers.

        Snippets are included newest-relevance-first until the character budget is exhausted, so
        a large repo degrades by dropping the least relevant source rather than dropping all code.
        """
        parts = [
            f"User Question: {query}",
            f"Active Feature Context: {current_feature or 'none established yet'}",
        ]
        if existing_summary:
            parts.append(f"Earlier In This Conversation: {existing_summary}")

        # Placed before the snippets deliberately: these are the statements the model should
        # trust when the two sources disagree, and the snippet block can be long enough that
        # anything after it competes for attention.
        if graph_facts:
            parts.append(graph_facts)

        if not sources:
            parts.append(
                "\nRetrieved Code Context: NONE. No code in this repository matched the question. "
                + ("The static analysis facts above still stand and should be used. "
                   if graph_facts else
                   "Tell the user you could not find it and ask them to name a file or symbol. ")
                + "Do not guess at file names."
            )
            return "\n".join(parts)

        parts.append("\nRetrieved Code Context (verbatim from the indexed repository):")

        # The remainder of the prompt budget, after the system prompt, the facts and the history
        # have taken their shares — never more than the standing per-request cap.
        #
        # This was a flat 12,000 characters regardless of what else was in the prompt, so adding
        # the facts block (G-01) raised every prompt by up to 2,900 tokens with nothing to
        # compensate. The cap was sized when it was the only thing in the context.
        # Everything already committed: this context so far, the system prompt, and the history
        # allowance. Previously the system prompt was not counted at all.
        spent = len("\n".join(parts)) + reserved_chars
        remaining = cls._char_budget() - spent - int(
            cls._char_budget() * cls.HISTORY_BUDGET_SHARE)
        budget = max(1500, min(cls.MAX_CODE_CHARS_TOTAL, remaining))
        included = 0

        for s in sources:
            snippet = (s.get("code_snippet") or "").strip()
            header = f"\n--- {s['file_path']} lines {s['start_line']}-{s['end_line']} (symbol: {s['symbol']}, type: {s['entity_type']}) ---"

            if not snippet:
                parts.append(header + "\n(no source captured for this entity)")
                continue

            if len(snippet) > cls.MAX_CODE_CHARS_PER_SOURCE:
                snippet = snippet[: cls.MAX_CODE_CHARS_PER_SOURCE] + "\n... (snippet truncated)"

            if len(snippet) > budget:
                remaining = len(sources) - included
                parts.append(f"\n({remaining} further matched source(s) omitted to stay within the context budget.)")
                break

            fence = cls._FENCE_LANG.get(s.get("language", ""), "")
            parts.append(f"{header}\n```{fence}\n{snippet}\n```")
            budget -= len(snippet)
            included += 1

        return "\n".join(parts)

    @classmethod
    async def stream_llm_provider(
        cls,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[str]:
        """
        Yields content deltas from the provider as they arrive.

        Both providers speak the OpenAI server-sent-events format, so one reader handles each.
        Any failure simply ends the stream; the caller falls back to the rule-based answer, the
        same contract as the buffered path.
        """
        provider = cls._resolve_provider()
        if not provider:
            return

        payload = {
            "model": provider["model"],
            "messages": cls._build_messages(system_prompt, user_prompt, history),
            "temperature": 0.3,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {provider['key']}",
            "Content-Type": "application/json",
            "User-Agent": "CodebaseIntelligence/1.0",
        }

        try:
            client = await get_http_client()
            async with client.stream("POST", provider["url"], json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    logger.error("%s stream returned HTTP %s: %s", provider["name"], response.status_code, body[:300])
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    try:
                        choices = json.loads(data).get("choices") or []
                        delta = choices[0].get("delta", {}).get("content") if choices else None
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta

        except httpx.TimeoutException:
            logger.error("%s stream timed out after %ss", provider["name"], cls.LLM_TIMEOUT_SECONDS)
        except httpx.HTTPError as e:
            logger.error("%s stream transport error: %s", provider["name"], e)

    @classmethod
    async def stream_user_query(cls, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        Runs a query and yields events: one `meta`, then `token` deltas, then `done`.

        Retrieval, tracing and impact analysis all complete before the first token, so the UI can
        render sources and the execution flow immediately instead of waiting on the full answer.
        """
        prepared = await cls.prepare_query(**kwargs)

        yield {
            "type": "meta",
            "intent": prepared["intent"],
            "sources": prepared["sources"],
        }

        collected: List[str] = []
        if settings.GROQ_API_KEY or settings.OPENAI_API_KEY:
            async for delta in cls.stream_llm_provider(
                prepared["sys_prompt"], prepared["ctx_text"], prepared["history"]
            ):
                collected.append(delta)
                yield {"type": "token", "text": delta}

        answer = "".join(collected)
        if not answer:
            # No LLM configured, or the stream failed. Emit the rule-based answer as one token so
            # the client renders identically either way.
            answer = cls.fallback_answer(prepared)
            yield {"type": "token", "text": answer}

        result = await cls._finish(prepared, answer)
        yield {"type": "done", "result": result}

    @classmethod
    async def prepare_query(cls, **kwargs) -> Dict[str, Any]:
        """Runs everything up to generation. See process_user_query for the parameters."""
        return await cls.process_user_query(_prepare_only=True, **kwargs)

    @classmethod
    async def process_user_query(
        cls,
        repository_id: str,
        conversation_id: str,
        query: str,
        all_chunks: List[Dict[str, Any]],
        existing_summary: str = "",
        working_context: Dict[str, Any] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        _prepare_only: bool = False
    ) -> Dict[str, Any]:
        working_context = working_context or {}
        history = history or []

        # Step 1: Intent Classification
        intent_info = QueryIntentClassifier.classify(query)
        intent = intent_info["intent"]

        # Step 2: Hybrid Retrieval
        retrieved_chunks = HybridRetriever.retrieve(
            repository_id=repository_id,
            query=query,
            all_chunks=all_chunks,
            top_k=6
        )

        # Resolution order for the subject of the question:
        #   1. a symbol the user named explicitly
        #   2. the conversation's current subject, when the query refers back to it ("who calls it?")
        #   3. the top-ranked retrieval hit
        target_feature = cls._extract_target_feature(query, repository_id, all_chunks)
        carried_symbol = working_context.get("current_symbol", "")

        target_symbol = cls._extract_target_symbol(query, all_chunks)
        # A file named in the query only takes over when no symbol was named: "what calls
        # `parse` in decide.py" is still a question about `parse`.
        target_file = "" if target_symbol else cls._extract_target_file(query, all_chunks)
        # Whether the user actually indicated a subject, as opposed to one being guessed from the
        # top retrieval hit. The guess is fine for ranking, but stating "these are ALL the callers
        # of X" about a symbol nobody mentioned is a confident answer to an unasked question.
        symbol_is_explicit = bool(target_symbol)
        if not target_symbol and carried_symbol and cls._is_followup_reference(query):
            target_symbol = carried_symbol
            symbol_is_explicit = True
        if not target_symbol and retrieved_chunks:
            target_symbol = retrieved_chunks[0].get("symbol", "")

        # No hardcoded default feature: an unknown feature must stay empty so downstream
        # formatters say "I don't know" rather than inventing one from the sample repo.
        current_feature = target_feature or working_context.get("current_feature", "")
        current_symbol = target_symbol or working_context.get("current_symbol", "")

        updated_working_ctx = {
            "current_feature": current_feature,
            "current_symbol": current_symbol,
            "relevant_files": list(set([c["file_path"] for c in retrieved_chunks]))
        }

        # Step 3: Extract sources
        sources = []
        for chunk in retrieved_chunks:
            sources.append({
                "file_path": chunk["file_path"],
                "symbol": chunk["symbol"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "entity_type": chunk["entity_type"],
                "language": chunk.get("language", ""),
                "code_snippet": chunk["code_snippet"]
            })

        execution_flow = {}
        impact_analysis = {}

        if intent == "FEATURE_EXPLANATION" and current_feature:
            execution_flow = FeatureTracer.trace_feature_flow(current_feature, all_chunks)
        elif intent == "CHANGE_IMPACT":
            target_sym = current_symbol or (retrieved_chunks[0]["symbol"] if retrieved_chunks else "")
            if target_sym:
                impact_analysis = ChangeImpactAnalyzer.analyze_change_impact(target_sym, all_chunks)

        # Step 4: System Prompt - Conversational & Progressive Detail
        sys_prompt = (
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

        # For a dependency or impact question the subject *is* the question, so the best
        # available guess is worth using. For anything else, only a subject the user actually
        # indicated earns an authoritative block about it — otherwise a greeting picks up the
        # complete caller list of whatever happened to rank first.
        subject_for_facts = current_symbol if (
            symbol_is_explicit or intent in ("DEPENDENCY_ANALYSIS", "CHANGE_IMPACT")
        ) else ""

        # When the question is about a file, the inferred symbol is noise — it is whatever ranked
        # first, which is exactly what used to produce an impact report about a test helper.
        if target_file:
            subject_for_facts = ""

        graph_facts = cls._build_graph_facts(
            intent, subject_for_facts, repository_id, execution_flow, impact_analysis,
            target_file=target_file, all_chunks=all_chunks,
        )
        # `MAX_GRAPH_FACTS` caps each block; this caps the sum of them. On a helper called from
        # 120 files the three blocks that qualify produced 11,684 characters between them —
        # roughly 2,900 tokens of a 8,000-token-per-minute budget, before any source code.
        budget = cls._char_budget()
        graph_facts = cls._fit_to_budget(
            graph_facts, int(budget * cls.FACTS_BUDGET_SHARE), "static analysis facts")
        ctx_text = cls._build_context(
            query, current_feature, existing_summary, sources, graph_facts,
            reserved_chars=len(sys_prompt),
        )

        # Everything up to here is shared with the streaming path; `_finish` below turns a raw
        # LLM response (or its absence) into the final payload.
        prepared = {
            "conversation_id": conversation_id,
            "repository_id": repository_id,
            "intent": intent,
            "query": query,
            "sys_prompt": sys_prompt,
            "ctx_text": ctx_text,
            "history": history,
            "existing_summary": existing_summary,
            "sources": sources,
            "execution_flow": execution_flow,
            "impact_analysis": impact_analysis,
            "working_context": updated_working_ctx,
            "current_feature": current_feature,
            "current_symbol": current_symbol,
            "retrieved_chunks": retrieved_chunks,
        }

        if _prepare_only:
            return prepared

        # Step 5: Execute LLM reasoning or fallback
        llm_response = ""
        if settings.GROQ_API_KEY or settings.OPENAI_API_KEY:
            llm_response = await cls.call_llm_provider(sys_prompt, ctx_text, history=history)

        return await cls._finish(prepared, llm_response)

    @classmethod
    def fallback_answer(cls, prepared: Dict[str, Any]) -> str:
        """Renders the rule-based answer for a prepared query, used when no LLM output arrives."""
        intent = prepared["intent"]
        sources = prepared["sources"]

        if intent == "FEATURE_EXPLANATION":
            return cls._format_conversational_feature(
                prepared["current_feature"], prepared["execution_flow"], sources
            )
        if intent == "CHANGE_IMPACT":
            return cls._format_conversational_impact(prepared["impact_analysis"])
        if intent == "DEPENDENCY_ANALYSIS":
            chunks = prepared["retrieved_chunks"]
            target_sym = prepared["current_symbol"] or (chunks[0]["symbol"] if chunks else "")
            return cls._format_conversational_dependency(
                target_sym,
                neo4j_client.get_callers(target_sym, prepared.get("repository_id")),
                sources,
            )
        if intent == "CODE_LOCATION":
            return cls._format_conversational_location(prepared["query"], sources)
        return cls._format_conversational_general(prepared["query"], intent, sources)

    @classmethod
    async def _finish(cls, prepared: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """
        Assembles the response payload from a prepared query and whatever the LLM produced.

        When the provider produced nothing — rate-limited, unreachable, or not configured — the
        rule-based template answers instead. That template is genuinely useful: it cites real
        files and line ranges from the retrieved chunks. What it is not is *equivalent*, and
        returning it in a payload indistinguishable from a generated answer left the user with a
        visibly poorer response and nothing indicating why.

        `degraded` says which one this is, so a client can show it rather than leaving the reader
        to wonder whether the system simply got worse.
        """
        degraded = not llm_response
        answer_markdown = llm_response or cls.fallback_answer(prepared)

        updated_summary = await ConversationSummarizer.update(
            existing_summary=prepared["existing_summary"],
            messages=prepared["history"],
            user_msg=prepared["query"],
            assistant_msg=answer_markdown,
            working_ctx=prepared["working_context"],
            llm_call=cls.call_llm_provider if (settings.GROQ_API_KEY or settings.OPENAI_API_KEY) else None,
        )

        return {
            "conversation_id": prepared["conversation_id"],
            "repository_id": prepared["repository_id"],
            "intent": prepared["intent"],
            "answer": answer_markdown,
            # True when the answer came from the rule-based template rather than a model.
            "degraded": degraded,
            "degraded_reason": (
                "The language model could not be reached (it may be rate-limited). "
                "This answer was assembled directly from the indexed code — the files and line "
                "ranges are accurate, but the explanation is not written for your question."
            ) if degraded else None,
            "sources": prepared["sources"],
            # Returned whenever the intent produced them. These are already gated by intent at
            # computation time above, so no further filtering is needed — the previous substring
            # check on the raw query silently discarded a correctly-computed trace whenever the
            # user phrased the question without the literal words "flow"/"step"/"impact".
            "execution_flow": prepared["execution_flow"],
            "impact_analysis": prepared["impact_analysis"],
            "working_context": prepared["working_context"],
            "updated_summary": updated_summary
        }

    # ------------------------------------------------------------------
    # Rule-based fallback formatters.
    #
    # These run only when no LLM is configured or the LLM call fails. They must render
    # STRICTLY from their arguments: never name a file, symbol or test that was not passed
    # in. Inventing a plausible-looking default turns a "no data" case into a confident
    # wrong answer, which is far worse for a tool developers use to learn a codebase.
    # ------------------------------------------------------------------

    NO_MATCH = (
        "I couldn't find anything matching that in this repository's index. "
        "Try naming a specific file, function or class — or re-index the repository if it was just uploaded."
    )

    @classmethod
    def _format_conversational_feature(cls, feature: str, flow: Dict[str, Any], sources: List[Dict[str, Any]]) -> str:
        steps = (flow or {}).get("flow_steps", [])

        if not steps and not sources:
            return cls.NO_MATCH

        label = f"**{feature}**" if feature else "That"
        lines = []

        if steps:
            lines.append(f"{label} runs through {len(steps)} step(s) across the layers I can see:\n")
            for st in steps[:8]:
                lines.append(f"{st['step']}. **{st['layer']}** — `{st['title']}` in `{st['file_path']}` (lines {st['lines']})")
            if len(steps) > 8:
                lines.append(f"\n...and {len(steps) - 8} further step(s).")
        else:
            files = ", ".join(f"`{s['file_path']}`" for s in sources[:3])
            lines.append(
                f"I couldn't assemble an ordered execution flow for {label.lower()}, but the closest matching code is in {files}."
            )

        lines.append("\n💡 *Want the full step list, the exact line ranges, or the change-impact view for any of these?*")
        return "\n".join(lines)

    @classmethod
    def _format_conversational_impact(cls, impact: Dict[str, Any]) -> str:
        if not impact:
            return cls.NO_MATCH

        target = impact.get("primary_target", {})
        callers = impact.get("confirmed_callers", [])
        tests = impact.get("affected_tests", [])
        files = impact.get("confirmed_files", [])

        symbol = target.get("symbol") or "that symbol"
        path = target.get("file_path") or "an unknown file"
        lines = [f"Changing `{symbol}` in `{path}` (lines {target.get('lines', '?')}) affects the following."]

        if callers:
            lines.append(f"\n**Confirmed callers ({len(callers)}):**")
            for c in callers[:6]:
                loc = f" — `{c['file_path']}`" if c.get("file_path") else ""
                lines.append(f"- `{c['symbol']}`{loc}")
            if len(callers) > 6:
                lines.append(f"- ...and {len(callers) - 6} more")
        else:
            lines.append("\nNo callers were found in the call graph — it may be an entry point, or only called dynamically.")

        if tests:
            lines.append(f"\n**Tests covering it ({len(tests)}):**")
            for t in tests[:6]:
                lines.append(f"- `{t['test_symbol']}` in `{t['file_path']}` (lines {t['lines']})")
        else:
            lines.append("\n⚠️ No tests reference this symbol — changes here are currently uncovered.")

        inferred = impact.get("inferred_impacts", [])
        if inferred:
            lines.append(f"\n**Likely affected indirectly ({len(inferred)}):**")
            for item in inferred[:5]:
                confidence = item.get("confidence", "medium")
                lines.append(f"- `{item['symbol']}` — {item['reason']} _({confidence} confidence)_")

        if files:
            lines.append(f"\n**Files touched:** {len(files)}")

        lines.append("\n💡 *Want caller signatures, the line-by-line implementation, or the dependency graph for this symbol?*")
        return "\n".join(lines)

    @classmethod
    def _format_conversational_dependency(cls, symbol: str, callers: List[Dict], sources: List[Dict]) -> str:
        if not symbol and not sources:
            return cls.NO_MATCH

        defined_in = f"`{sources[0]['file_path']}`" if sources else None
        lines = []

        if callers:
            names = ", ".join(f"`{c['caller_symbol']}`" for c in callers[:5])
            more = f" (+{len(callers) - 5} more)" if len(callers) > 5 else ""
            lines.append(f"`{symbol}` is called by {names}{more}.")
        else:
            lines.append(
                f"I found no callers of `{symbol}` in the call graph. "
                "It may be an entry point, unused, or invoked dynamically in a way static analysis can't see."
            )

        if defined_in:
            lines.append(f"\nIt's defined in {defined_in}.")

        lines.append("\n💡 *Want its callees, the files that import it, or caller line numbers?*")
        return "\n".join(lines)

    @classmethod
    def _format_conversational_location(cls, query: str, sources: List[Dict]) -> str:
        if not sources:
            return cls.NO_MATCH

        src = sources[0]
        lines = [
            f"That looks like `{src['symbol']}` in `{src['file_path']}` (lines {src['start_line']}-{src['end_line']})."
        ]

        others = sources[1:4]
        if others:
            lines.append("\nOther possible matches:")
            for s in others:
                lines.append(f"- `{s['symbol']}` — `{s['file_path']}` (lines {s['start_line']}-{s['end_line']})")

        lines.append("\n💡 *Want the code itself, its callers, or the impact of changing it?*")
        return "\n".join(lines)

    @classmethod
    def _format_conversational_general(cls, query: str, intent: str, sources: List[Dict]) -> str:
        if not sources:
            return cls.NO_MATCH

        lines = ["Here's the most relevant code I found for that:\n"]
        for s in sources[:4]:
            lines.append(f"- `{s['symbol']}` ({s['entity_type']}) — `{s['file_path']}` lines {s['start_line']}-{s['end_line']}")

        lines.append("\n💡 *Ask for line numbers, an execution flow, or a dependency breakdown on any of these.*")
        return "\n".join(lines)
