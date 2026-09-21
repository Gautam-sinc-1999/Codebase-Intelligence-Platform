# GitHub integration — implementation plan (phases 1 & 2)

Adding "paste a GitHub URL", "pick a branch", and "tell me when the branch has moved on"
alongside the existing zip upload.
Commit-level browsing is deliberately **out of scope** here — see [Why commits are deferred](#why-commits-are-deferred).

**Companions:** [fix.md](fix.md) (48 audit issues, 45 fixed) · [fix2.md](fix2.md) (14 end-to-end issues, open)

---

## What this is, in one line

The zip path is `extract → directory → index`. Git is `clone → directory → index`.
**Only the acquisition step changes.** `index_repository_folder(dir, repo_id, name, force_full)`
takes a plain directory and does not care how it got there.

---

## Measured assumptions

Everything below is timed on this machine against a real public repository
(`psf/requests`, 128 files), not estimated.

| Operation | Cost | Notes |
|---|---|---|
| `git ls-remote --heads <url>` | **1.2 s** | No clone, **no auth for public repos** |
| `git clone --depth 1 --single-branch` | **2.2 s**, 7.7 MB | History not needed |
| Indexing + embedding (857 chunks) | **~23 s**, 89 % embedding | From the RETAIL end-to-end run |

**Acquisition is cheap; embedding dominates.** That single fact drives most decisions here:
the work is not "get the code", it is "do not re-embed what we already have", and "do not
block an HTTP request for 25 seconds".

---

## Blockers to clear first

Two open issues become blocking rather than merely untidy once branches exist.

### B-1 — Indexing is synchronous ([G-11](fix2.md#g-11))

Upload is a blocking HTTP request; the RETAIL upload took **36 s**. Clone-and-index is the same
shape. A browser or proxy will time out on a large repository, and the user gets no progress.

**Required:** return `202 Accepted` with a `repository_id` immediately, index in a background
worker, and let the client poll. `indexing_status` and `GET /repositories/{id}/status` already
exist — they just are not used this way yet.

**This is a prerequisite, not an optimisation.** Phase 1 is not shippable without it.

### B-2 — There is no way to delete a repository ([G-10](fix2.md#g-10))

No `DELETE /repositories/{id}` exists, and nothing cleans up the Chroma collection, the Neo4j
nodes, the persisted index or the extracted source.

With zip uploads this was untidy. **With branches it is unbounded growth**: every branch is a
full index — its own Chroma collection, its own graph — and branches are created and deleted
constantly. Ten branches on one repository is ten times the storage and embedding cost, none of
which can ever be reclaimed.

**Required before phase 2**, not after.

---

## Phase 1 — Index from a GitHub URL

### Identity

Today `repository_id` is `repo_<uuid4>` — a new id per upload, so uploading the same project
twice creates two unrelated repositories. For git, identity should be **derived**, so re-syncing
the same repository updates it rather than duplicating it:

```
repository_id = "gh_" + sha256("github.com/owner/repo@branch")[:12]
```

Stable across syncs, distinct per branch, and it keeps the existing `repository_id` shape so
nothing downstream changes.

Store alongside the existing record: `source` (`zip` | `github`), `clone_url`, `owner`, `repo`,
`branch`, `commit_sha`, `synced_at`.

### API

```
POST /api/repositories/from-github
     { "url": "https://github.com/owner/repo", "branch": "main" }
  -> 202 { repository_id, name, branch, status: "indexing" }

GET  /api/repositories/{id}/status        # already exists
POST /api/repositories/{id}/sync          # re-clone and incrementally re-index
DELETE /api/repositories/{id}             # B-2
```

`POST /sync` is `POST /reindex` with a fresh clone in front of it. The incremental path
([F-23](fix.md#f-23)) then does its job: files whose content hash is unchanged keep their chunks,
and only changed files are re-parsed and re-embedded.

### Acquisition

```
git clone --depth 1 --single-branch --branch <branch> <url> <tmpdir>
```

Shallow and single-branch: history is never used, and a full clone of a large repository is
mostly history. Capture `git rev-parse HEAD` as `commit_sha` for display and for later sync
comparison.

### Input validation — treat a URL as hostile input

A clone is remote-controlled disk writing, exactly like the zip path, and deserves the same
scrutiny that [F-03](fix.md#f-03) and [F-40](fix2.md#g-40) applied there.

- **Accept only** `https://github.com/<owner>/<repo>` (plus `.git`). Reject `git://`, `ssh://`,
  `file://` and anything with a `..` segment — `file:///etc` is a valid clone URL.
- **Reject `--upload-pack` and option-looking inputs.** A URL beginning with `-` can be read as a
  git flag; pass `--` before the URL.
- **Apply the existing limits to the clone**, not only to zips: `MAX_ARCHIVE_ENTRIES`,
  `MAX_TOTAL_UNCOMPRESSED_BYTES`, `MAX_INDEXED_FILE_BYTES`. A repository can be a zip bomb.
- **Timeout the clone** (`GIT_HTTP_LOW_SPEED_LIMIT` / `timeout`) so a slow remote cannot pin a worker.
- **Reuse `_is_ignorable_archive_entry`-style filtering** so `.git/`, `node_modules/` and secret
  files are skipped before indexing. Note the clone *does* contain `.git/`, which discovery
  already excludes — but the disk cost is real, so delete the working tree after indexing.
- **Do not keep the source.** Index, persist the index, delete the clone. The index store already
  holds everything retrieval needs, and not storing customer source removes a liability.

**Deleting the clone breaks `/reindex` for git repositories — this must be handled, not
discovered.** `POST /{id}/reindex` reads `storage_path` and returns **409** when the directory is
gone, which is correct for a zip (the source really is lost) but wrong for a clone (the source is
one `git clone` away). The sidebar's ⟳ Re-index button would fail on every GitHub repository.

Resolution: `/reindex` and `/sync` stay separate endpoints, and the client picks by
`source` — `zip` → `/reindex`, `github` → `/sync`. `/reindex` additionally returns a more useful
409 for git repositories, pointing at `/sync` rather than telling the user to re-upload.

### Private repositories

Out of scope for phase 1 — public URLs only. When it arrives, prefer a **GitHub App**
(short-lived per-installation tokens, fine-grained scopes) over storing PATs. Note that a token
in a clone URL leaks into `ps` output and error messages; use a credential helper or
`http.extraHeader`.

### Frontend

`RepoUploader.jsx` currently takes a single `.zip` file input. Becomes a two-tab modal:

```
┌ Add repository ─────────────────┐
│ [ Upload .zip ] [ GitHub URL ]  │
│                                 │
│  https://github.com/owner/repo  │
│  [ Fetch branches ]             │
└─────────────────────────────────┘
```

New in `services/api.js`: `fetchGithubBranches(url)`, `importFromGithub(url, branch)`,
`fetchRepositoryStatus(id)`. Because indexing is now async, the modal closes immediately and the
sidebar shows the repository with an "indexing…" state driven by polling the status endpoint.

---

## Phase 2 — Branch listing and selection

### Listing

```
GET /api/repositories/github/branches?url=...
 -> { default_branch, branches: [{ name, commit_sha, is_default }] }
```

Backed by `git ls-remote --heads` — **1.2 s, no clone, no token for public repos**. The GitHub
REST API is an alternative but needs a token to avoid a 60/hour rate limit, so `ls-remote` is
strictly better here.

Determining the default branch: `git ls-remote --symref <url> HEAD`.

**Cache per URL for a few minutes.** The UI will call this on every paste, and branch lists do
not change by the second.

### The cost model, stated plainly

**Every branch is a full index.** For a RETAIL-sized repository that is ~857 chunks, ~20 s of
embedding and ~32 MB of Chroma storage *per branch*. Ten branches is ten times all of it.

Mitigations, in order of value:

1. **Index on demand.** Show all branches; index only the one the user selects.
2. **Default to the default branch.** Most users want `main`.
3. **Evict.** Least-recently-used branch indexes are dropped after N days — which requires B-2.
4. **Cap per repository** (say 5 branches), with an explicit message rather than silent failure.

Content-addressed chunks would let branches *share* embeddings, since branches of the same
repository are ~99 % identical. That is the phase-3 design and is noted below — but phase 2
should not wait for it.

### UI

```
Branches in owner/repo:
  ● main          (default) — indexed 2h ago
  ○ develop                  — not indexed
  ○ feat/checkout            — not indexed
[ Sync selected branch ]
```

Each indexed branch is a separate entry in the existing sidebar, labelled `owner/repo@branch`.

---

## Phase 2.5 — Repository drift and thread staleness

A thread is a conversation *about a specific version of the code*. Upstream keeps moving. When a
user reopens a thread a week later, the answers in it may describe code that no longer exists.

The user-facing behaviour: on reopening a thread, detect that the branch has moved and offer a
choice — keep this version, or pull the latest and re-index.

### The check

`git ls-remote <url> refs/heads/<branch>` returns the current remote SHA in **1.08 s** (measured
on this machine, no auth). Compare it to the `commit_sha` stored on the repository record.

**Do not block thread opening on it.** Messages render from local disk in milliseconds; the drift
check is a network call. Fire it after the thread renders and let the banner appear when the
answer arrives. Cache per `(url, branch)` for 5 minutes — the same cache phase 2 needs for branch
listing.

```
GET /api/repositories/{id}/drift
 -> { branch, current_sha, remote_sha, behind: true, checked_at }
```

Separate from `GET /status` deliberately: status is polled during indexing and must stay fast and
local; drift costs a second and hits the network.

For `source: "zip"` repositories there is no upstream — return `behind: false` and skip the check
entirely rather than inventing a comparison.

### Three states, not two

> **As built:** a sync now writes a marker into every affected thread *and advances that thread's
> recorded commit*, so a thread that has been told is no longer reported as `thread_behind`. The
> marker sits at the exact version boundary in the transcript, which is a better place for it
> than a banner at the top. `thread_behind` remains for a thread the index moved under without
> any marker being written — a zip re-index, a restored thread, a marking that failed.

The obvious comparison is "repo vs remote". There is a third case that appears as soon as one
repository has more than one thread:

| `thread.indexed_commit_sha` | `repo.commit_sha` | `remote_sha` | Meaning | Action |
|---|---|---|---|---|
| = | = | = | Everything current | Nothing |
| = | = | **≠** | **Upstream moved** | Offer sync — the case in the brief |
| **≠** | = | = | **Another thread already synced this repo** | Inform only; there is nothing to pull |

The third row matters: sync a repository from one thread and every *other* thread on that
repository is now discussing a version that is no longer indexed — without anything upstream
having changed. Those threads need a different message ("this thread's answers describe an
earlier version"), not a sync button that would do nothing.

So `thread.json` gains two fields:

- `indexed_commit_sha` — the repository's `commit_sha` when this thread's last turn was answered
- `drift_ack_sha` — the remote SHA the user explicitly chose to stay behind, so the banner is
  shown once per new commit rather than on every message

### The consequence nobody expects: citations age

Persisted turns store citation **pointers** — `(file_path, start_line, end_line)` — not code
bodies, and `GET /{id}/snippet` resolves them against the *current* index ([F-49](fix.md)).

**After a sync, old pointers resolve against new code.** A citation recorded as
`checkout.py:40-58` still resolves, but line 40 may now be a different function. The viewer would
confidently show the wrong code, which is worse than showing nothing.

Rejected fix: pinning each thread to its own index (an index per commit). The commit analysis
below prices that at ~428,500 chunks and ~16 GB for a 500-commit repository.

**Adopted fix — label, do not pin.** Each assistant turn already knows the SHA it was answered
at. When resolving a citation, compare the turn's SHA with the index's current SHA and mark the
result:

```
GET /{id}/snippet?...&at_sha=a1b2c3d
 -> { ..., "stale": true, "indexed_sha": "f9e8d7c" }
```

The viewer then shows the code with an explicit "from a newer version of this file — this answer
was written against `a1b2c3d`" note. Honest, costs nothing, and needs no extra storage.

This is cheap now and becomes exact under the content-addressed design below, where a pointer can
name the precise chunk version it cited.

### Resolution

**Keep this version** — `POST /api/conversations/{id}/ack-drift { sha }` records `drift_ack_sha`.
No indexing, no cost. The banner stays quiet until the branch moves again.

**Pull latest** — `POST /api/repositories/{id}/sync`: re-clone, incremental re-index, update
`commit_sha` and `synced_at`. Only changed files re-embed ([F-23](fix.md#f-23)); on an 885-chunk
repository a one-file edit left 876 chunks untouched, so a sync is seconds, not the full ~23 s.

Append a marker message to the thread recording the change:

```
— synced to f9e8d7c · 14 files changed · answers above describe a1b2c3d —
```

The transcript then shows its own version boundary, which is what makes the stale-citation label
comprehensible instead of mysterious.

### UI

```
┌──────────────────────────────────────────────────────┐
│ ⟳ owner/repo@main has moved since this thread began. │
│   Indexed a1b2c3d · latest f9e8d7c                   │
│   [ Keep this version ]  [ Pull latest & re-index ]  │
└──────────────────────────────────────────────────────┘
```

Non-blocking: the thread is fully usable while the banner is showing and while a sync runs.

---

## Work breakdown

**Status: all items implemented.** 320 backend tests pass on both graph backends (Neo4j and the
NetworkX fallback), plus 9 frontend tests; the zip upload path is unchanged throughout and has a
test asserting so.

| # | Task | Depends on | Size | Done |
|---|---|---|---|---|
| 1 | **B-1** async indexing: job runner, `202`, status polling | — | M | ✅ |
| 2 | **B-2** `DELETE /repositories/{id}` + full cleanup (Mongo, Chroma, Neo4j, index, source) | — | M | ✅ |
| 3 | `GitSource` module: validate URL, shallow clone, capture SHA, enforce limits, clean up | — | S | ✅ |
| 4 | `POST /from-github` + derived `repository_id` + git metadata on the record | 1, 3 | S | ✅ |
| 5 | `POST /{id}/sync` — re-clone, incremental re-index | 4 | S | ✅ |
| 6 | Frontend: two-tab modal, async status polling in the sidebar | 1, 4 | M | ✅ |
| 7 | `GET /github/branches` via `ls-remote`, with caching | 3 | S | ✅ |
| 8 | Frontend: branch picker | 6, 7 | S | ✅ |
| 9 | Branch caps + LRU eviction | 2, 8 | S | ⬜ deferred |
| 10 | `GET /{id}/drift` + the 5-minute `ls-remote` cache | 3, 7 | S | ✅ |
| 11 | `indexed_commit_sha` / `drift_ack_sha` on `thread.json`; stamp each turn | 4 | S | ✅ |
| 12 | `POST /conversations/{id}/ack-drift`; sync marker message | 11 | S | ✅ |
| 13 | Stale-citation labelling on `/snippet` (`at_sha` → `stale`) | 11 | S | ✅ |
| 14 | Frontend: drift banner, non-blocking check on thread open | 10, 12 | M | ✅ |
| 15 | Tests (see below) | all | M | ✅ |

Phase 1 is items 1–6; phase 2 is 7–9; phase 2.5 is 10–14.

---

## Tests to add

The suite is at 133 tests across both graph backends; these extend it. Pattern the fixtures on
`tests/conftest.py`, which isolates `DATA_DIR`, `CHROMADB_DIR` and the Mongo database.

**Do not hit the network in unit tests.** Create a local git repository in `tmp_path`
(`git init`, commit, branch) and clone from the filesystem — real git, no flakiness, no rate limits.

- URL validation: reject `file://`, `git://`, `ssh://`, `..` segments, leading `-`
- Branch listing against a local repo with several branches
- Import → index → the four query types answer correctly
- Same URL + branch imported twice yields **one** repository, not two
- Same URL, two branches yields **two** repositories with **isolated graphs**
  (the [F-45](fix.md#f-45) regression, now reachable by ordinary use)
- Sync after a commit picks up the change and re-embeds only the changed files
- `DELETE` removes the Mongo record, the Chroma collection, the Neo4j nodes and the index file
- Clone limits: a repository exceeding the entry/size caps is rejected
- The working tree is deleted after indexing

Phase 2.5:

- Drift is **not** reported when the local SHA equals the remote SHA
- Drift **is** reported after committing to the local origin repo (commit, then re-check)
- `source: "zip"` repositories always report `behind: false` and make no network call
- `drift_ack_sha` suppresses the banner for that SHA, and stops suppressing on the next commit
- The three-way case: two threads on one repository, sync from thread A, thread B reports
  "answers describe an earlier version" and offers **no** sync button
- A citation recorded before a sync is returned with `stale: true` after it
- A citation whose symbol is unchanged across the sync is **not** marked stale
- Sync appends the marker message, and `indexed_commit_sha` advances only for the synced thread
- The drift check never blocks: a thread with an unreachable remote still opens and answers

---

## Why commits are deferred

Indexing every commit independently does not scale — for a 500-commit repository that is roughly
**428,500 chunks, ~2.8 hours of embedding and ~16 GB** of vector storage.

The measurement that makes it tractable later: on an 885-chunk repository, editing one file left
**876/885 chunks byte-identical**. A typical commit changes well under 1 % of chunks.

So the phase-3 design is **content-addressed chunks** — `chunk_id = hash(symbol + code)` rather
than today's `hash(repo:file:symbol:start_line)`. One embedding per unique *version* of a
function, shared by every commit and every branch containing it; a commit becomes a cheap
manifest of chunk ids. That turns ~428,500 chunks into ~5,300.

Two notes for when it is picked up:

- **It improves today's behaviour too.** Because `start_line` is part of the id, adding a comment
  at the top of a file changes the id of every symbol below it and forces a re-embed. Content
  addressing makes incremental re-index genuinely incremental — and would make branches share
  storage, which phase 2 currently cannot.
- **Worth validating the requirement first.** "Browse the repo at commit X" is expensive; *"what
  changed in this commit and what does it affect?"* is a diff plus the impact analysis that
  already exists, costs almost nothing, and may be the thing users actually want.

---

## Open questions

1. **Public only for phase 1?** Private repositories mean token storage and a GitHub App — real
   security surface, and better as its own phase.
2. **What happens to zip upload?** Keep both, or migrate? Suggest keeping it: it is the only path
   for code not in GitHub.
3. **Auto-sync or manual?** This plan is manual — a Sync button, plus the phase-2.5 prompt when a
   thread is reopened onto a moved branch. Webhook-driven auto-sync is a natural follow-on and
   reuses all of it; note that auto-sync would make stale citations the normal case rather than
   the exception, which raises the value of labelling them.
4. **Branch cap?** Suggest 5 per repository initially, with eviction.
5. **Should a sync ever be refused?** If the branch was force-pushed or rebased, the old
   `commit_sha` is no longer an ancestor of the new one and *most* citations will move. Detectable
   with `git merge-base --is-ancestor`. Suggest syncing anyway but warning more loudly, since the
   alternative — refusing — leaves the user stuck on a commit that no longer exists upstream.
