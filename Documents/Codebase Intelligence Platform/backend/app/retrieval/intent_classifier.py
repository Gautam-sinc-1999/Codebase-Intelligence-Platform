import re
from typing import Dict, Any, List, Tuple

class QueryIntentClassifier:
    """
    Classifies user prompt intent to select optimal retrieval and reasoning strategy.

    **Matching is on whole words, not substrings (G-02).** The previous implementation asked
    `if any(w in q_lower for w in [...])`, which fires on any occurrence anywhere inside a word:

        "Which parts are unchanged since last release?"  -> CHANGE_IMPACT        ('unchanged')
        "Is the parser independent of the lexer?"        -> DEPENDENCY_ANALYSIS  ('independent')
        "Describe the workflow engine"                   -> FEATURE_EXPLANATION  ('workflow')

    Every one of those routes retrieval down the wrong path before a single chunk is fetched.
    Inflections are therefore listed explicitly rather than caught by prefix matching: a prefix
    rule would reject 'unchanged' and 'exchange' correctly but still accept 'changelog' and
    'changepoint', and being precise about a few dozen words is cheaper than being subtly wrong.

    **Scoring replaces first-match-wins.** The old cascade returned on the first list that
    matched, so intent depended on the order the `if` statements happened to be written in —
    "If I change the discount logic, who calls it?" was CHANGE_IMPACT purely because that branch
    came first. Now every intent is scored and the best one wins, with the declaration order used
    only to break exact ties.
    """

    INTENTS = [
        "FEATURE_EXPLANATION",
        "CODE_LOCATION",
        "DEPENDENCY_ANALYSIS",
        "CHANGE_IMPACT",
        "DEBUGGING",
        "ARCHITECTURE"
    ]

    # Ordered by priority: an exact score tie resolves to whichever appears first here, which
    # preserves the precedence the original cascade encoded.
    #
    # Multi-word entries are matched as consecutive token sequences and score their own length,
    # so "who calls" is stronger evidence than a lone "flow" — that is the point of scoring.
    PATTERNS: List[Tuple[str, List[str], float]] = [
        ("CHANGE_IMPACT", [
            "change", "changes", "changed", "changing",
            "modify", "modifies", "modified", "modifying",
            "affect", "affects", "affected",
            "break", "breaks", "breaking",
            "impact", "impacts", "impacted",
            "if i edit", "if i change", "what breaks", "knock on effect", "side effects",
        ], 0.95),

        ("DEPENDENCY_ANALYSIS", [
            "depend", "depends", "depended", "depending",
            "dependency", "dependencies", "dependent", "dependents",
            "who calls", "called by", "used by", "calls into", "consumers of",
            "import", "imports", "imported",
            "caller", "callers",
        ], 0.92),

        ("CODE_LOCATION", [
            "where is", "where are", "where does", "where can i find",
            "locate", "located",
            "find file", "file for", "which file", "what file",
            "which line", "what line", "line number",
        ], 0.90),

        ("FEATURE_EXPLANATION", [
            "how does", "how do", "how is", "how are",
            "explain feature", "walk me through", "what happens when",
            "flow", "flows", "lifecycle", "end to end",
        ], 0.88),

        ("DEBUGGING", [
            "why", "fail", "fails", "failed", "failing", "failure",
            "error", "errors", "bug", "bugs", "issue", "issues",
            "crash", "crashes", "crashed", "exception", "traceback",
            "not working", "broken", "stack trace",
        ], 0.85),

        ("ARCHITECTURE", [
            "architecture", "architectural", "overview", "structure", "structured",
            "component", "components", "stack", "layout",
            "high level", "big picture", "design of",
        ], 0.80),
    ]

    DEFAULT_INTENT = "FEATURE_EXPLANATION"
    DEFAULT_CONFIDENCE = 0.70

    _TOKEN_RE = re.compile(r"[a-z0-9_]+")

    # A token immediately followed by one of these reads as a filename, not a keyword. The
    # repository behind this issue contains a module literally named `impact.py`, and "what does
    # impact.py do?" is a question about a file, not a change-impact analysis.
    _FILE_EXTENSIONS = frozenset({
        "py", "js", "ts", "jsx", "tsx", "java", "go", "rb", "php", "cs", "cpp", "c", "h",
        "json", "yaml", "yml", "toml", "md", "txt", "sql", "html", "css", "sh",
    })

    @classmethod
    def _tokenize(cls, query: str) -> List[str]:
        """
        Lowercased word tokens, with filename stems dropped.

        Splitting on non-word characters means `impact.py` yields ('impact', 'py'); the extension
        check then removes the stem so it cannot be read as the keyword 'impact'.
        """
        raw = cls._TOKEN_RE.findall(query.lower())
        tokens: List[str] = []
        for i, token in enumerate(raw):
            following = raw[i + 1] if i + 1 < len(raw) else None
            if following in cls._FILE_EXTENSIONS:
                continue
            tokens.append(token)
        return tokens

    @staticmethod
    def _count_matches(tokens: List[str], pattern_tokens: List[str]) -> int:
        """Occurrences of a consecutive token sequence within the query's tokens."""
        span = len(pattern_tokens)
        if span == 0 or span > len(tokens):
            return 0
        return sum(
            1 for i in range(len(tokens) - span + 1)
            if tokens[i:i + span] == pattern_tokens
        )

    @classmethod
    def classify(cls, query: str) -> Dict[str, Any]:
        tokens = cls._tokenize(query or "")
        if not tokens:
            return {"intent": cls.DEFAULT_INTENT, "confidence": cls.DEFAULT_CONFIDENCE, "score": 0}

        best_intent = cls.DEFAULT_INTENT
        best_score = 0
        best_confidence = cls.DEFAULT_CONFIDENCE

        for intent, patterns, confidence in cls.PATTERNS:
            score = 0
            for pattern in patterns:
                pattern_tokens = pattern.split()
                hits = cls._count_matches(tokens, pattern_tokens)
                # A phrase scores its own length, so a two-word signal outweighs a one-word one.
                score += hits * len(pattern_tokens)

            # Strictly greater: the first intent to reach a given score keeps it, which is how
            # the original cascade's precedence is preserved for ties.
            if score > best_score:
                best_intent, best_score, best_confidence = intent, score, confidence

        if best_score == 0:
            return {"intent": cls.DEFAULT_INTENT, "confidence": cls.DEFAULT_CONFIDENCE, "score": 0}

        return {"intent": best_intent, "confidence": best_confidence, "score": best_score}
