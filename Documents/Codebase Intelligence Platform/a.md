
gautamkoshta@Gautams-MacBook-Air ~ % brew services stop redis 
==> Downloading Homebrew API data
✔︎ JSON API packages.arm64_tahoe.jws.json            Downloaded   15.5MB/ 15.5MB
Stopping `redis`... (might take a while)
==> Successfully stopped `redis` (label: sh.brew.redis)
gautamkoshta@Gautams-MacBook-Air ~ % brew services stop mongodb-community@8.0
Stopping `mongodb-community@8.0`... (might take a while)
==> Successfully stopped `mongodb-community@8.0` (label: homebrew.mxcl.mongodb-c
gautamkoshta@Gautams-MacBook-Air ~ % brew services stop neo4j
Stopping `neo4j`... (might take a while)
==> Successfully stopped `neo4j` (label: homebrew.mxcl.neo4j)
gautamkoshta@Gautams-MacBook-Air ~ % 


brew services start redis 
brew services start mongodb-community@8.0
brew services start neo4j

brew services stop redis 
brew services stop mongodb-community@8.0
brew services stop neo4j



1. Document types
Not documents — source code, which changes every decision downstream. Repositories arrive two ways: a zip upload, or a GitHub clone by URL and branch.

Parsing is per-language: Python via the stdlib ast, JavaScript/TypeScript/JSX/TSX and Java via tree-sitter, SQL via regex. The output isn't text — it's structured entities carrying symbol, parent_symbol, calls, imports, route, http_calls, and exact start/end lines.

2. Chunking
Symbol-level and AST-aware — deliberately not a token window. One chunk is one function, method or class, with its real line range preserved.

The reason is citations. The product's core promise is "checkout starts in checkout_controller.py:40-58", and a fixed-size window chunk cannot cite — it starts and ends mid-function at an offset that means nothing to a reader. Symbol boundaries make every chunk independently quotable.

Two details that earned their keep:

The embedded text isn't raw code. It's Symbol: … / File: path:start-end / Language: … / <code>. Putting the symbol name and path into the embedded string gives both the vector and the keyword index something to bite on — a question naming calculate_discount matches lexically even when the body doesn't repeat the name.
chunk_id = md5(repo:file:symbol:start_line) — content-derived and stable. That's what makes incremental re-indexing possible: on a sync, files whose SHA-256 is unchanged keep their chunks, and only changed files are re-parsed and re-embedded.
3. Embedding and vector store
all-MiniLM-L6-v2, 384-dim, ONNX runtime, local. ChromaDB, persistent, one collection per repository.

Be honest about the tradeoff if asked: a code-specific encoder would almost certainly embed code better. MiniLM was chosen because it's fast, free, runs locally with no API dependency — and because the architecture doesn't lean on it for precision. Exact symbol matching is BM25's job. That's a deliberate division of labour, not an oversight.

Collection-per-repository matters operationally: deletion is a collection drop rather than a filtered delete, and cross-repository leakage is impossible by construction rather than by query hygiene.

Cost reality: indexing psf/requests — 109 files → 865 chunks, 947 graph nodes — takes 41 seconds, ~89% of it embedding. That single number drove the whole async job design: uploads return 202 and index in the background, because a browser or proxy abandons the request long before it finishes.

4. Retrieval — hybrid, then reranked
Vector search and BM25 run in parallel and fuse (vector hit +2.5, BM25 score × 0.1), then a code-specific reranker with six signals:

Signal	Weight
Entity kind (function/class/etc.)	multiplicative, so it shapes without swamping
Exact symbol match	+4.0 — strongest evidence the developer named this thing
Query-term coverage in the body	+1.5 × coverage
Path relevance (excluding generic dirs like src, api)	+1.0
Span penalty — stubs ≤2 lines / whole files >400	−0.5 / −1.0
Test files, unless the question is about tests	−1.0
The bug worth telling: query tokens were split on whitespace, so "calculate_discount?" kept its question mark and never matched the symbol calculate_discount. Lexical search was silently failing on ordinary questions and results leaned entirely on the vector store — with no error anywhere. Fixed by tokenising on [A-Za-z0-9_]+. It's a good story because the system looked like it worked.

The part that isn't RAG — and the most interesting failure
Retrieval returns top-k. For a widely-used helper, the call graph has every call site while the snippets show a handful. So a Neo4j call graph (NetworkX fallback) is queried and injected as "Static Analysis Facts" alongside the snippets, with an explicit count and completeness claim.

The model then did something instructive: it averaged the two sources. Thirteen callers in the facts, five in the snippets, and it wrote "it's called in a few places, for example…". Two fixes, and you need both:

Prompt: the facts are declared authoritative and complete, with the specific hedging words — "partial", "examples", "a handful" — explicitly forbidden.
Measurement: graph_recall / graph_precision, so the fix is verified rather than assumed.
How I measured answer quality — three layers
Layer 1 — deterministic, free, on every query. No judge model, no added latency:

citation_validity (do cited files exist in the index) · citation_line_validity (do the line ranges exist — this catches stale citations after a repo syncs) · path_grounding (is every path in the prose one the model was actually shown) · graph_recall / graph_precision · retrieval_hit · prompt_budget_used · degraded.

The principle I'd emphasise: a metric that can't be computed is omitted, never defaulted to zero. Defaulting makes the dashboard average an average over queries the metric was meaningless for — which is worse than having no number.

Layer 2 — RAGAS-style, offline, on a dataset. Faithfulness, answer relevancy, context precision, context entity recall, noise sensitivity. Two design points:

It never runs in the request path. Faithfulness alone decomposes an answer into claims and checks each — 15–30 judge calls per item, against a product making one call per query and already rate-limited.
The judge is a separate provider and key (Gemini free tier) so evaluation never competes with a user waiting for an answer.
context_precision is rank-weighted average precision specifically because it's the metric that scores the reranker — retrieving the right chunk at position 6 is a worse run than at position 1, and a plain mean can't tell them apart.

Layer 3 — human. Thumbs up/down on each answer → user_feedback, scored 1/0 so the average reads as a satisfaction rate, attached to that answer's own trace. This is the only signal that measures whether the answer was useful — every layer-1 score can be perfect for an answer that didn't help.

All of it traced in Langfuse, with datasets seeded from existing test fixtures so a regression run is reproducible, and prompt versions linked to generations so scores group by prompt version: change rule A1 → run the dataset → faithfulness 0.91 → 0.78 → roll the label back to v3.

If they push on weaknesses, these are the real ones — better to name them first: the embedding model isn't code-specific; fusion weights and reranker signals are hand-tuned, not learned; there are no reference answers, so the RAGAS variants used are the reference-free ones; and the RAGAS metrics are faithful implementations of the published definitions rather than the ragas library's output, so they detect movement but aren't comparable to published benchmarks.




Hi [Name],

I built a **code-focused RAG pipeline** designed for repository-level question answering, where the input is source code rather than conventional documents.

**1. Document types & parsing**
Repositories were provided either through ZIP uploads or GitHub URLs/branches. I used language-specific parsers: Python `ast`, Tree-sitter for JavaScript/TypeScript/JSX/TSX and Java, and regex for SQL. Instead of converting code into plain text, I extracted structured entities such as symbols, parent symbols, calls, imports, routes, HTTP calls, and exact line ranges.

**2. Chunking**
I used **AST/symbol-level chunking** rather than fixed token windows. Each chunk represented a function, method, or class with its actual file and line range. This was important for accurate citations.

The embedded text followed a structure like:

`Symbol → File path:start-end → Language → Code`

I also generated stable chunk IDs using:

`md5(repo:file:symbol:start_line)`

This enabled incremental re-indexing, where only changed files were re-parsed and re-embedded.

**3. Embeddings & vector database**
I used **all-MiniLM-L6-v2 (384 dimensions)** with ONNX Runtime for local embeddings and **ChromaDB** as the persistent vector database, with one collection per repository.

MiniLM was selected for its speed, local execution, and lack of API dependency. Since code-specific semantic precision was limited, I complemented vector retrieval with lexical BM25 retrieval.

**4. Retrieval & reranking**
The pipeline used **hybrid retrieval**, running vector search and BM25 in parallel and fusing the results. A code-specific reranker then considered signals such as:

* Exact symbol match
* Entity type/function/class relevance
* Query-term coverage
* File/path relevance
* Chunk-size/span penalties
* Test-file penalties when the query wasn't test-related

One useful debugging discovery was that whitespace tokenization caused queries like `calculate_discount?` to miss the exact symbol match. I fixed this using regex-based tokenization: `[A-Za-z0-9_]+`.

**5. Beyond traditional RAG – call graph integration**
For code-understanding questions, I also integrated a **Neo4j call graph**, with NetworkX as a fallback. Graph-derived facts were injected alongside retrieved snippets as explicit "Static Analysis Facts."

This helped answer questions such as *"How many places call this function?"* more reliably than relying only on retrieved snippets.

**6. Answer quality measurement**
I measured quality at three levels:

* **Online deterministic metrics:** citation validity, citation line validity, path grounding, retrieval hit rate, graph recall/precision, prompt-budget usage, and degraded-state tracking.
* **Offline RAG evaluation:** RAGAS-style metrics including faithfulness, answer relevancy, context precision, context entity recall, and noise sensitivity.
* **Human feedback:** thumbs up/down feedback attached to individual traces to measure actual answer usefulness.

The evaluation pipeline ran offline rather than in the production request path. I used a separate evaluation provider/key to avoid affecting production latency.

All traces, evaluations, and prompt versions were tracked through **Langfuse**, allowing reproducible regression testing and comparison across prompt versions.

The main limitations I identified were that MiniLM is not code-specific, retrieval/reranking weights were hand-tuned rather than learned, and the evaluation dataset did not contain reference answers. I treated these as known trade-offs rather than assuming the evaluation metrics were equivalent to published benchmarks.

Best regards,
Gautam Koshta
