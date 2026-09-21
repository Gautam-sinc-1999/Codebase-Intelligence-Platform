# Codebase Intelligence Platform

Conversational RAG for unfamiliar codebases. Upload a zip or paste a GitHub URL — it parses every
symbol, embeds them, and builds a call graph across the frontend/backend boundary. Ask questions in
plain English and get answers cited to real line numbers.

The part that distinguishes it from embedding files into a chat window: **static analysis feeds the
prompt**. Retrieval returns the top *k* chunks, so for a helper used in fifty places it surfaces a
handful and a model will report those as the complete set. The call graph knows all fifty, and that
list is injected as authoritative fact.

```
"What depends on calculate_discount?"

  → DEPENDENCY_ANALYSIS
  → 4 callers, from the graph — with the count stated in the answer
  → each cited as file:line, resolvable back to source on click
```

**Languages** Python · JavaScript · TypeScript · JSX · TSX · Java
**Frameworks** FastAPI · Flask · Express route detection, including Blueprint and APIRouter prefixes

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Starting the backing services](#starting-the-backing-services)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Verifying it works](#verifying-it-works)
- [Using it](#using-it)
- [Tests](#tests)
- [API reference](#api-reference)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Further reading](#further-reading)

---

## Requirements

| | Version used | Notes |
|---|---|---|
| Python | 3.13 | 3.10+ should work; 3.13 is what this is developed against |
| Node | 26.x | For the frontend only; 18+ is enough |
| Docker | any recent | Optional — the easiest way to get the three services |
| git | 2.x | Required for GitHub import (`clone`, `ls-remote`) |

The three backing services are **MongoDB**, **Redis** and **Neo4j**. The app runs without any of
them — every store has an in-process fallback — but then it is single-process and the graph
fallback has different performance characteristics from Cypher. See
[Running without the services](#running-without-the-services).

---

## Quick start

From the project root:

```bash
# 1. backing services
docker compose up -d

# 2. Python dependencies
python3 -m venv venv
./venv/bin/pip install -r backend/requirements.txt

# 3. configuration — see the Configuration section
cp backend/.env.example backend/.env    # then add your LLM key

# 4. backend
cd backend && ../venv/bin/uvicorn app.main:app --reload --port 8000

# 5. frontend, in a second terminal
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000**. Vite proxies `/api` to the backend on port 8000, so both need to
be running.

---

## Starting the backing services

### With Docker (recommended)

`docker-compose.yml` at the project root defines all three, each bound to `127.0.0.1` so nothing is
exposed on the network.

```bash
docker compose up -d           # start
docker compose ps              # check
docker compose logs -f neo4j   # follow one
docker compose down            # stop, keeping data
docker compose down -v         # stop and delete the volumes
```

Neo4j takes about 40 seconds on a cold start. The Neo4j browser is at **http://localhost:7474**
(user `neo4j`, password `password`).

### With Homebrew (macOS)

```bash
brew services start redis
brew services start neo4j
brew services start mongodb-community@8.0
```

> **MongoDB may not start this way.** On some installs `brew services` reports
> `Formula 'mongodb-community@8.0' has not implemented #plist, #service or provided a locatable
> service file` — it looks for `sh.brew.…plist` while the install ships `homebrew.mxcl.…plist`.
> Load the plist directly instead:
>
> ```bash
> launchctl bootstrap gui/$(id -u) \
>   /opt/homebrew/opt/mongodb-community@8.0/homebrew.mxcl.mongodb-community@8.0.plist
> ```
>
> Confirm with `nc -z localhost 27017 && echo up`.

### Checking they are all reachable

```bash
nc -z localhost 27017 && echo "mongo  up"
nc -z localhost 6379  && echo "redis  up"
nc -z localhost 7687  && echo "neo4j  up"
```

### Running without the services

The app starts and works with none of them. Fallbacks:

| Service | Fallback | Consequence |
|---|---|---|
| MongoDB | JSON files under `data/fallback_db` | Durable, single-process only |
| Redis | in-memory dict | Working context lost on restart |
| Neo4j | in-process NetworkX graph | Rebuilt per process from the chunk index |
| ChromaDB | in-memory vectors | Embeddings recomputed on restart |

`GET /` reports which backend each store actually resolved to, so degraded operation is visible
rather than silent.

---

## Configuration

Configuration lives in **`backend/.env`**. It is git-ignored, and the indexer treats any `.env` as a
secret — it is never read into an index or written to disk during extraction.

```ini
# ── LLM (required for generated answers) ──────────────────────
LLM_PROVIDER=auto                 # auto | groq | openai
GROQ_API_KEY=gsk_...              # https://console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1
# OPENAI_API_KEY=sk-...           # used when LLM_PROVIDER=openai or as fallback
# DEFAULT_LLM_MODEL=gpt-4o

# ── Datastores ────────────────────────────────────────────────
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=Codebase_Intelligence
REDIS_URI=redis://localhost:6379
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# ── Storage paths ─────────────────────────────────────────────
DATA_DIR=./data
CHROMADB_DIR=./data/chroma_db

# ── Limits ────────────────────────────────────────────────────
MAX_UPLOAD_BYTES=262144000        # 250 MB
MAX_ARCHIVE_ENTRIES=20000
MAX_INDEXED_FILE_BYTES=2097152    # 2 MB
MAX_PROMPT_TOKENS=5000            # budget for the whole prompt

# ── Security (optional) ───────────────────────────────────────
# API_KEY=                        # set to require X-API-Key on every route
# CORS_ORIGINS=http://localhost:3000
```

**Without an LLM key** the system still works: every answer is assembled from the retrieved chunks
by a rule-based template, cites real files and line ranges, and is flagged `degraded: true` in the
response so the UI can say so. Retrieval, the graph and citations are unaffected.

---

## Running the app

### Backend

```bash
cd backend
../venv/bin/uvicorn app.main:app --reload --port 8000
```

On startup it will:

1. connect to each store, logging which backend it resolved to
2. index the bundled `sample_repo` if it is not already indexed
3. migrate any folder-per-thread conversations into MongoDB (idempotent, never deletes the folders)
4. drop repository records whose index *and* source are both gone

### Frontend

```bash
cd frontend
npm install       # first time only
npm run dev       # http://localhost:3000
```

`npm run build` produces `dist/`, which the backend does not serve — use the dev server, or serve
`dist/` behind your own static host.

---

## Verifying it works

```bash
curl -s localhost:8000/ | python3 -m json.tool
```

```json
{
  "status": "online",
  "backends": {
    "documents": "mongodb-motor",
    "cache": "redis",
    "vectors": "chromadb",
    "graph": "neo4j"
  },
  "degraded": []
}
```

`degraded: []` means all four stores are live. Anything listed there is running on a fallback, and
`degraded_reasons` distinguishes *driver not installed* from *server unreachable* — they have
completely different fixes.

Then ask the bundled sample repository a question:

```bash
CONV=$(curl -s -X POST localhost:8000/api/conversations \
  -H 'Content-Type: application/json' \
  -d '{"repository_id":"sample_ecommerce_repo","title":"smoke test"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["conversation_id"])')

curl -s -X POST localhost:8000/api/conversations/$CONV/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"What depends on calculate_discount?"}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["intent"]); print(d["answer"][:400])'
```

Expect intent `DEPENDENCY_ANALYSIS` and an answer naming four callers.

---

## Using it

### Upload a zip

Sidebar → **+ Upload Zip** → pick a `.zip`. The request returns as soon as the archive is
extracted; indexing continues in the background and the sidebar shows progress. A corrupt or
oversized archive fails immediately with a clear message.

### Import from GitHub

Sidebar → **+ Upload Zip** → the **🐙 GitHub URL** tab.

1. paste `https://github.com/owner/repo` and click **Branches** (~1 s, no clone)
2. pick a branch — already-indexed branches are marked
3. **Import & Index**

Only the selected branch is cloned, shallow (`--depth 1`), and the working copy is deleted once
indexed. Public repositories only.

Each branch is its own repository entry, so switching branches means switching entries in the
dropdown. Their indexes and graphs are fully isolated.

### Keeping a repository current

The **⟳ Sync** button (GitHub repos) re-clones and re-indexes only what changed. **⟳ Re-index**
(zip repos) re-reads the stored source. Shift-click either for a full rebuild.

When you reopen a conversation whose branch has moved on, a banner offers **Keep this version** or
**Pull latest & re-index**. Citations written before a sync are labelled as coming from an older
commit rather than silently showing whatever now occupies those lines.

### Managing conversations

Double-click a thread name, or click **✎**, to rename it. **✕** deletes it. Deleting a repository
keeps its conversations by default and tells you how many are affected.

---

## Tests

```bash
cd backend

# against Neo4j
../venv/bin/python -m pytest tests/ -q

# against the in-memory graph fallback
NEO4J_URI="bolt://127.0.0.1:9" ../venv/bin/python -m pytest tests/ -q

# frontend
cd ../frontend && npm test
```

**Run both graph backends.** It is not redundancy — running both is what catches the two quietly
disagreeing, which has happened three times (label casing, whether the queried symbol appears in its
own dependencies, and whether `max_depth` was honoured at all).

Tests use an isolated `DATA_DIR`, `CHROMADB_DIR` and MongoDB database, and clean up after
themselves. Neo4j Community allows only one user database, so graph isolation works by recording
every repository id written and deleting exactly those — never by an allowlist.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Liveness, and which backend each store resolved to |
| `POST` | `/api/repositories/upload` | Upload a zip — **202**, indexes in the background |
| `GET` | `/api/repositories` | List repositories |
| `GET` | `/api/repositories/github/branches?url=` | Branches, without cloning |
| `POST` | `/api/repositories/from-github` | Import a branch — **202** |
| `GET` | `/api/repositories/{id}` | One repository |
| `GET` | `/api/repositories/{id}/status` | Indexing state, chunk count, commit |
| `GET` | `/api/repositories/{id}/drift` | Has the branch moved on? |
| `POST` | `/api/repositories/{id}/sync` | Re-clone and re-index (GitHub) |
| `POST` | `/api/repositories/{id}/reindex` | Re-index from stored source (zip) |
| `GET` | `/api/repositories/{id}/snippet` | Resolve a citation back to source |
| `DELETE` | `/api/repositories/{id}` | Delete, with optional conversation cascade |
| `POST` | `/api/conversations` | Create a thread |
| `GET` | `/api/conversations` | List threads |
| `GET` | `/api/conversations/{id}` | One thread with its messages |
| `PATCH` | `/api/conversations/{id}` | Rename |
| `DELETE` | `/api/conversations/{id}` | Delete |
| `POST` | `/api/conversations/{id}/messages` | Ask a question |
| `POST` | `/api/conversations/{id}/messages/stream` | Same, as server-sent events |
| `POST` | `/api/conversations/{id}/ack-drift` | Stay on the current version |
| `GET` | `/api/graph/{id}` | Graph data, with truncation reported |

`{id}` above is shorthand — the actual parameters are `{repository_id}` and `{conversation_id}`.
Interactive docs, with exact names and schemas, at **http://localhost:8000/docs**.

Set `API_KEY` in `.env` to require `X-API-Key` on every route; set `VITE_API_KEY` in the frontend
environment to match.

---

## Project layout

```
backend/app/
  api/           repositories · github · conversations · graph
  agents/        orchestrator (intent → retrieve → facts → prompt) · impact_analyzer
  ingestion/     discovery · ast_parser · treesitter_parser · chunker
                 git_source · index_store
  retrieval/     hybrid_retriever · reranker · intent_classifier
  graph/         neo4j_client (+ NetworkX fallback) · feature_tracer
  memory/        mongo_thread_store · conversation_memory · summarizer
  vector/        chromadb_client
  core/          config · database · jobs · security

frontend/src/
  components/    Sidebar · ChatWindow · MessageItem · DependencyGraph
                 RepoUploader · DriftBanner
  services/      api.js
  lib/           drift.js

data/            indexes/ · chroma_db/ · repositories/   (git-ignored)
```

---

## Troubleshooting

**`degraded: ["cache"]` but Redis is running**
The `redis` package is probably not installed. The log distinguishes this from an unreachable
server. Run `./venv/bin/pip install -r backend/requirements.txt`.

**MongoDB will not start with `brew services`**
See [the note above](#with-homebrew-macos) — load the plist with `launchctl` directly.

**MongoDB crashed and will not restart**
Check free disk space first. MongoDB shuts down when it cannot write, and the failure appears in
`/opt/homebrew/var/log/mongodb/mongo.log` as a fatal error in `ftdc`.

**A repository shows as indexing forever**
Check `GET /api/repositories/{id}/status`. A failed import deletes its repository record but keeps
the failure readable there, including the reason.

**`⟳ Re-index` returns 409 on a GitHub repository**
Expected. A GitHub import keeps no local source — the clone is deleted after indexing — so use
**Sync**, which re-clones first. The UI picks the right one automatically.

**Answers are worse than expected and mention being unavailable**
The response carries `degraded: true`: no LLM was reachable, so the answer came from the rule-based
template. Check `GROQ_API_KEY`, or that you are not rate-limited.

**Tests leave large directories in `/tmp`**
They should not — the suite removes its temp root and sweeps roots older than an hour. If you find
`ci_tests_*` directories after an interrupted run, they are safe to delete.

---

## Further reading

| File | Contents |
|---|---|
| `architecture.html` | Full internals — indexing pipeline, call graph, retrieval, the real prompt and a traced query with measured timings |
| `github-integration-plan.md` | Design and build log for GitHub import, branch selection and drift detection |
| `implementation_plan.md` | Original build plan |
