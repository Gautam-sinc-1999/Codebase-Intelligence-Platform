# langfuse_plan.md — observability and evaluation

Adding per-query tracing and automated scoring to the Codebase Intelligence Platform.

**Companions:** [README.md](README.md) · [architecture.html](architecture.html) ·
[fix2.md](fix2.md) (14 of 15 fixed) · [github-integration-plan.md](github-integration-plan.md)

**Status:** plan only. Nothing implemented.

---

## The problem, stated plainly

Two things are missing, and they are different problems.

**1. No trace.** When an answer is wrong there is no way to see *why* — which chunks were
retrieved, what they scored, whether the target symbol was even in the top six, how large the
prompt was, whether the graph facts were truncated. All of that is computed and then discarded.

**2. No regression check.** There is currently no way to tell whether a prompt change made answers
better or worse. The `5/13 → 13/13` improvement was measured by hand, once. A later change could
silently undo it and nothing would notice.

The second is the more serious. The system has 520 tests covering behaviour, and **zero** covering
answer quality.

---

## Tool choice — Langfuse

### Why

| Reason | Detail |
|---|---|
| **No LangChain here** | The orchestrator is a hand-written sequential function. LangSmith's main advantage is auto-instrumentation of LangChain; without it you write manual spans anyway, so pick the tool best at manual spans. Langfuse is framework-agnostic by design. |
| **Self-hostable** | `docker-compose.yml` already runs three services. A fourth is natural, costs nothing, and keeps traces of private source code on the machine. |
| **Tracing *and* scores in one place** | That is the actual requirement. Most tools do one well. |
| **Datasets and experiments** | The regression check above is exactly what dataset runs are for. |

### The alternative worth knowing

**Arize Phoenix** — also open source and local, and stronger out of the box on RAG-specific
evaluators (retrieval relevance, hallucination). If *scoring* matters more than production tracing,
Phoenix is the better pick. If both, Langfuse.

Ruled out: **Helicone** (proxy-level — cannot see the retrieval stages), **raw OpenTelemetry +
Jaeger** (all the LLM semantics would have to be built by hand), **LangSmith** (best value is tied
to a framework not in use here).

---

## Trace structure

One trace per query, seven child spans. The boundaries already exist in
`CodebaseAgentOrchestrator.process_user_query` — they simply are not recorded.

Measured on a real query (`"What depends on calculate_discount?"`, `sample_ecommerce_repo`):

| Span | Time | Attributes worth capturing |
|---|---|---|
| `load_index` | 47.8 ms | chunk count, cache hit/miss |
| `classify_intent` | 0.1 ms | intent, confidence, score |
| `retrieve` | 350.4 ms | **per-chunk fusion score, rerank score, which signal contributed** |
| `resolve_subject` | 0.1 ms | resolved symbol, whether explicit or inferred, target file |
| `graph_facts` | 9.6 ms | caller count, dependency count, whether truncated |
| `build_prompt` | see note | chars/tokens by section: system, facts, history, code |
| `llm` | 2,577.9 ms | model, input/output tokens, cost, retries, degraded |

> **A correction, and the reason this plan exists.** `build_prompt` was first recorded at 835.1 ms.
> That figure is wrong: the measuring script called retrieval standalone and *then* called
> `process_user_query(_prepare_only=True)`, which runs retrieval again internally — so the span
> included a duplicate retrieval pass. The true cost is far lower and is not yet known.
>
> The other six timings stand. But it is a clean illustration of the problem: a script measuring
> from outside the system mismeasured the system. Instrumentation inside the code path is what
> stops that.

**The LLM is roughly 68% of wall clock.** Everything the system does to make the answer correct
costs on the order of a second combined. Worth remembering before optimising anything other than
the prompt.

### The query path is wider than these seven spans

Reading `api/conversations.py` shows work on both sides of the orchestrator that the list above
misses entirely:

| Gap | Why it matters |
|---|---|
| `_load_conversation_context` | Loads the thread from MongoDB, then calls `get_repository_chunks` — which **on a cache miss reads the whole index from disk and rebuilds the repository's Neo4j graph**. Potentially seconds, currently invisible. Also the origin of the 404 and 409 responses. |
| **`ConversationSummarizer.update`** | **A second LLM call per turn** (`summarizer.py:145`, gated on `should_summarize`). Untracked tokens and untracked cost. |
| `save_turn` | The write-through to MongoDB — two documents per turn |

The summarizer is the one to fix first. **A cost dashboard that misses an entire LLM call is not a
cost dashboard**, and every figure in the [Token cost](#token-cost) section below is understated
until it is traced.

### Sessions

`conversation_id` maps to a Langfuse **session**, so a multi-turn thread groups naturally and
scores can be watched across a conversation rather than only per query.

---

## The flows not traced at all

The query path is one of several. Some of the others are larger.

| Flow | Scale |
|---|---|
| **Indexing** | 6 stages, **41 s** for `psf/requests`, ~89% of it embedding. Arguably deserves tracing more than queries do — it is the slowest thing the system does. |
| **GitHub import** | `validate → clone → limits → index → cleanup` |
| **Sync + drift** | `ls-remote`, incremental re-index, reuse counts (the "1 re-parsed, 2 reused" signal) |
| **Streaming endpoint** | `/messages/stream` is a **separate code path** — instrument the buffered one and this is still dark |
| **Snippet resolution** | Every citation expand, including the `at_sha` staleness check |
| **Frontend** | Click → rendered. Server-side timing misses the proxy hop and the render. |

### Three trace types, not one

```
query_trace      load_context → [7 spans] → summarize → save_turn
                                            ^^^^^^^^^ the currently-missed LLM call

index_trace      discover → parse → chunk → embed → graph → persist
                 one per repository, linked to the background job id

acquire_trace    validate_url → ls-remote → clone → enforce_limits → cleanup
```

Plus the per-conversation **session** already described above, tying query traces together.

**Indexing deserves its own trace even though nobody watches it live.** It is a background job, so
latency does not annoy anyone directly — but it is where 89% of the compute goes, where incremental
re-index either works or silently does not, and where a repository can fail in ways that only show
up as a 409 later. The per-file reuse counts are exactly the kind of thing that is obvious in a
trace and invisible in a log.

---

## Scores

Three layers, distinguished by what they cost.

### Layer 1 — deterministic, every query, zero LLM calls

These are the ones to build first. No judge, no labels, no added latency.

| Score | Type | Computed from |
|---|---|---|
| `citation_validity` | 0–1 | % of cited `file:line` that resolve via `GET /repositories/{id}/snippet` |
| **`graph_recall`** | 0–1 | **callers named in the answer ÷ callers in the graph** |
| **`graph_precision`** | 0–1 | **callers named that are real ÷ callers named** — catches invented callers |
| `path_grounding` | 0–1 | % of file paths in the answer that appear in the retrieved context |
| `retrieval_hit` | bool | was the resolved target symbol in the top-6? |
| `prompt_budget_used` | 0–1 | tokens ÷ `MAX_PROMPT_TOKENS` — early warning before rate limits return |
| `facts_truncated` | bool | did the graph-facts block get cut to fit? |
| `degraded` | bool | template vs model — already a field on the response |

### The insight this system has and most RAG projects do not

**For dependency and change-impact queries, the call graph is an exact oracle.**

RAGAS `context_recall` asks an LLM to judge whether retrieval found everything needed. Here there is
nothing to judge: the graph knows every caller of a symbol, exactly. `graph_recall` is
deterministic, free, runs on every query, and has no judge that can be wrong.

This is the same measurement as the `5/13 → 13/13` result — the difference is that it would run
continuously instead of once by hand.

> **The interview line:** *"We use RAGAS for the generative metrics, but for retrieval completeness
> we have a deterministic oracle, so we don't depend on a judge for the thing that matters most."*

### Layer 2 — RAGAS, offline, on a dataset

| Metric | Measures | Fits? |
|---|---|---|
| **Faithfulness** | Every claim supported by the context | ✅ Core — catches invented behaviour |
| **Answer Relevancy** | Does it answer the question asked | ✅ Core |
| **Context Precision** | Are relevant chunks ranked high | ✅ Directly scores the fusion weights and reranker |
| **Context Entity Recall** | Ground-truth entities present in context | ✅ Entities map to symbol names |
| **Noise Sensitivity** | Does irrelevant context corrupt the answer | ✅ Relevant — both graph facts *and* snippets are injected |
| Context Recall | Did retrieval get everything needed | ⚠️ Superseded by `graph_recall` for the queries that matter |
| Answer Correctness | Vs a reference answer | ⚠️ Reference answers are expensive to author for code |
| Answer Semantic Similarity | Embedding distance to a reference | ⚠️ Weak signal for code answers |
| Aspect Critique | Custom binary judgments | ◐ Optional |
| Summarization · SQL · agentic (tool-call, goal accuracy, topic adherence) | — | ❌ Different problem shape |

### Layer 3 — human

A thumbs up/down in `MessageItem`, posted as a `user_feedback` score. One component change, and the
only signal that measures whether the answer was actually *useful*.

---

## The constraint that shapes everything

**Almost every RAGAS metric is LLM-as-judge.** Faithfulness decomposes an answer into claims and
checks each one — several calls. Six metrics over one query can mean 15–30 LLM calls.

This system is **already rate-limited on Groq with one call per query** — that was G-13, the last
open issue from the earlier audit. (`fix2.md` no longer exists on disk, so that link is dropped
rather than left dangling.) Running RAGAS inline would worsen the exact problem the prompt budget
was built to fix.

Therefore:

- **RAGAS never runs in the request path.** Offline, on a sampled dataset, nightly or per-PR.
- **The judge uses a separate provider and key** from the product, so evaluation never competes
  with answering for the same TPM budget.
- **Layer 1 carries the per-query signal.** It is free, so it can run on everything.

---

## Datasets and experiments

### Creating them

| Route | Use |
|---|---|
| UI — create and add items, or upload CSV/JSON | Curated cases |
| SDK | Generating from existing test fixtures |
| **From a trace** — "add to dataset" in the trace list | **The one that changes how you work** |

The third turns every bad answer noticed in the UI into a permanent regression case, which is
precisely what is missing today.

### Two datasets already exist

- **Intent classification — 18 labelled queries** from the G-02 work, with expected outputs
  (`"Which parts are unchanged?"` → must not be `CHANGE_IMPACT`). Drops in directly.
- **The 13-caller case** — a ready-made `graph_recall` item with a known-correct answer.

Most projects start observability with an empty dataset and no labels. This one starts populated.

### Running

Langfuse does **not** execute the application — the loop is yours: pull dataset items, call the
code, link each trace to a named run. Langfuse stores, links and compares runs side by side with
aggregated scores.

The workflow this enables: change the prompt → run the dataset → see `faithfulness` drop from 0.91
to 0.78 → revert. That check does not exist today.

### LLM-as-a-judge in Langfuse

Configure an evaluator with a model, a prompt template and a trace filter; it emits a named score
automatically. Templates ship for hallucination, relevance, correctness and toxicity, and custom
ones can be written.

> **Verify before designing around it.** Managed evaluator features have differed between Langfuse
> Cloud and self-hosted OSS, and some eval functionality has been cloud or enterprise-tier. Check
> what is available in the version being deployed.
>
> **Fallback that always works:** run RAGAS in a script and push each metric as a named score via
> the SDK. Works on any deployment and gives more control over the judge — and since RAGAS is wanted
> specifically, this may be the destination anyway.

---

## Token cost

Langfuse computes cost from the model name plus token counts, or it can be set explicitly.

Measured shape per query:

```
input    ~1,900 tokens   (1,739 system + 5,746 user chars ÷ 4)
output     ~200 tokens
model     llama-3.3-70b-versatile (Groq)
```

Track `input_tokens`, `output_tokens` and `total_cost` on the `llm` span. Langfuse then aggregates
per trace, session, user and day.

**This is not the whole bill.** The summarizer makes a second LLM call on turns where
`should_summarize` fires, and it is not in the figures above. Until that call is traced, every cost
number here is understated by an unknown amount — which is precisely the sort of thing that only
becomes visible once tracing exists.

**The useful figure is cost per conversation, not per query** — a thread replays history, so the
marginal cost of turn five is higher than turn one, and the summarizer fires more often as a thread
grows. Current per-token rates should be read from the provider; they change.

---

## Work breakdown

| # | Task | Depends on | Size | Status |
|---|---|---|---|---|
| 1 | Langfuse service in `docker-compose.yml`, bound to `127.0.0.1` | — | S | done |
| 2 | SDK wiring — trace per query, session per conversation | 1 | S | done |
| 3 | Seven spans with attributes, in `process_user_query` | 2 | M | done |
| 4 | **Extend the query trace to the full path** — `_load_conversation_context`, the summarizer's LLM call, `save_turn` | 3 | S | done |
| 5 | Token counts and cost on **both** LLM spans | 4 | S | done |
| 6 | Layer 1 scorers as a module (`app/observability/scorers.py`) | 3 | M | done |
| 7 | Wire Layer 1 scores onto each trace | 6 | S | done |
| 8 | Thumbs up/down in `MessageItem` → `user_feedback` score | 2 | S | done |
| 9 | **`index_trace`** — the six ingestion stages, linked to the job id | 2 | M | done |
| 10 | **`acquire_trace`** — clone, limits, cleanup | 9 | S | done |
| 11 | **Instrument the streaming endpoint** — a separate code path from the buffered one | 3 | S | done |
| 12 | Snippet resolution span, including the `at_sha` staleness check | 2 | S | done |
| 13 | Seed the two existing datasets | 1 | S | done — seeding into Langfuse not yet run |
| 14 | Dataset runner script (pull items → call code → link run) | 13 | M | done — `--local` verified; Langfuse-linked run not yet exercised |
| 15 | RAGAS harness with a separate judge provider | 14 | M | done — judge not yet run against a live key |
| 16 | Push RAGAS metrics back as scores | 15 | S | done — pushing not yet exercised |
| 17 | Tests for the scorers themselves | 6 | M | done |
| — | Frontend timing (click → rendered) | 2 | deferred | open |

### Notes from implementation

**Indexing and acquisition open their own root traces.** They run on worker threads dispatched by
`JobRunner` through `loop.run_in_executor`, which — unlike `asyncio.to_thread` — does not copy the
caller's context. A trace opened in the request would therefore never reach them. `job_id` and
`trigger` travel as trace metadata instead, which is what ties a trace back to the job a client is
polling. `JobRunner.submit` takes its own `job_id` and `fn` as positional-only parameters so that
forwarding a callee's `job_id` through `**kwargs` does not collide with the runner's own.

**A GitHub import produces two traces, not one**, because it has two lifetimes: `resolve_repository`
(URL validation and `ls-remote`, inside the request, before the 202) and `acquire` (clone, limits,
index, cleanup, on the worker thread afterwards). The index spans nest inside `acquire`.

**Still to confirm against a live Langfuse:** `start_trace_sync` calls `propagate_attributes` for
trace-level session/tags/metadata, and the nested index trace calls it again inside `acquire`. With
the recording fake this is exact; with the real SDK the inner call may overwrite the outer trace's
tags. Cosmetic if so, but it needs one look at a real trace to settle.

**Items 1–7 are the useful minimum**: the full query path traced end to end, both LLM calls costed,
and free per-query scores on everything.

**Items 9–12 close the remaining blind spots.** Item 9 is the one to prioritise among them — it is
where 89% of the compute goes. Item 11 is small but easy to forget, and forgetting it means half
the query traffic is untraced.

**Items 13–16 are the regression check.** Item 8 is an afternoon and gives the only signal that
measures whether an answer was actually useful.

---

## Honest caveats

**Layer 1 scores parse answer prose.** "Which callers did the model name?" means matching symbol
names against markdown — word-boundary regex. A model writing `` `process_checkout()` `` with
parentheses, or splitting a name across a table cell, can be missed. Good as a trend signal, not a
precise measurement. Worth spot-checking early against hand-counted results.

**Tracing adds a dependency to the request path.** The SDK batches and flushes in the background,
but a misconfigured or unreachable Langfuse must never fail a query. Whatever is built should follow
the pattern already used for every store here: fail open, log, and carry on.

**The SDK moves fast.** The scoring API has changed shape across versions. Check current docs rather
than any signature quoted from memory, including in this document.

**This adds a fifth service.** The stack is already broad — MongoDB, Redis, Neo4j, ChromaDB. That is
a fair thing for a reviewer to push on. The defence is that observability is developer-facing and
optional: the product runs without it, exactly as it runs without any of the other three.

---

## Open questions

1. **Self-hosted or Langfuse Cloud?** Self-hosted keeps traces of private source code local, which
   matters given what is being indexed. Cloud gets the managed evaluators without version caveats.
2. **Which judge model?** Needs to be a different provider from Groq to avoid competing for TPM.
   Cheapest capable option is usually right — the judge does not need to be the best model.
3. **How large should the eval dataset be?** Fifty queries is enough to detect a regression;
   two hundred to trust an absolute number. Start at fifty.
4. **Sample rate for RAGAS?** Every dataset run, certainly. On live traffic, probably not at all —
   Layer 1 already covers every query for free.
5. **Do Layer 1 scores gate anything?** A `graph_recall` below some threshold could fail CI on a
   prompt change. Tempting, but it needs a stable baseline first.
