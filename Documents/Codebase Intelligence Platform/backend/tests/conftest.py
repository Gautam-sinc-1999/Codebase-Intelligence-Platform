"""
Shared fixtures.

Every test runs against an isolated DATA_DIR and CHROMADB_DIR. This is not tidiness: the
verification scripts these tests were ported from set DATA_DIR but not CHROMADB_DIR, so they
wrote into the project's real vector store and left dozens of collections behind — and that
shared, pre-populated store masked two genuine defects (index/vector divergence, and punctuation
defeating lexical search) because semantic search always found *something*.

The environment must be set before any app module is imported, since Settings reads it at import.
"""
import os
import sys
import itertools
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Removed at the end of the session (see _remove_test_root). Left behind, each run leaked its
# own copy of the isolated DATA_DIR and CHROMADB_DIR — up to 38 MB apiece, since a session
# indexes dozens of repositories and writes real embeddings. Three hundred runs filled a disk.
#
# Stale roots from a crashed run are swept at startup too: teardown cannot run if the process
# dies, and that is precisely when nobody is watching.
_TEST_ROOT = tempfile.mkdtemp(prefix="ci_tests_")


def _sweep_abandoned_test_roots(keep: str) -> None:
    """Removes `ci_tests_*` roots left by runs that never reached teardown."""
    import shutil
    import time as _time

    parent = os.path.dirname(keep)
    cutoff = _time.time() - 3600  # an hour old: never touch a run happening right now
    for name in os.listdir(parent):
        if not name.startswith("ci_tests_"):
            continue
        path = os.path.join(parent, name)
        if path == keep:
            continue
        try:
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


try:
    _sweep_abandoned_test_roots(_TEST_ROOT)
except Exception:
    pass
os.environ["DATA_DIR"] = os.path.join(_TEST_ROOT, "data")
os.environ["CHROMADB_DIR"] = os.path.join(_TEST_ROOT, "chroma")

# Tracing off for the suite, regardless of what backend/.env holds.
#
# Once real Langfuse keys existed, every test began shipping traces to the live project: the
# run slowed by roughly 10x and the dashboard filled with hundreds of fixture traces, drowning
# the real ones it exists to show. Nothing is lost by disabling it — the tests that exercise
# tracing install their own recording client with monkeypatch and never consult these keys.
# Set to empty rather than deleted: `app.core.config` loads backend/.env with
# `os.environ.setdefault`, so a deleted key is simply restored the moment anything imports it —
# which `app.observability.tracing` does. An empty value survives, and reads as "not configured".
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

# A dedicated database, dropped at the end of the session.
#
# Isolating only the filesystem was not enough once MongoDB was running: repository records
# written by one run persisted into the next, pointing at tmp paths that no longer existed, and
# a re-index test then failed on stale state rather than on its own behaviour.
os.environ["MONGODB_DB_NAME"] = "codebase_intelligence_test"
os.environ.setdefault("API_KEY", "")
# Tests must exercise the rule-based paths deterministically, never a live provider.
os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402

settings.GROQ_API_KEY = ""
settings.OPENAI_API_KEY = ""

SAMPLE_REPO = str(BACKEND_ROOT.parent / "sample_repo")


@pytest.fixture(scope="session")
def sample_repo_path():
    return SAMPLE_REPO


@pytest.fixture(scope="session")
def sample_chunks():
    """The bundled sample repository, indexed once and shared across tests."""
    from app.api.repositories import index_repository_folder
    return index_repository_folder(SAMPLE_REPO, "test_sample", "sample")["all_chunks"]


# Session-wide, so repository ids are never reused. A per-fixture counter restarted at 1 in
# every test, so two tests both created "tmp_repo_1" — and since the graph is keyed by
# repository id, one test's symbols showed up in another's results.
_REPO_SEQUENCE = itertools.count(1)


@pytest.fixture
def temp_repo(tmp_path):
    """
    Builds a small repository on disk from a {relative_path: content} mapping and indexes it
    under a unique id, so tests that mutate files do not interfere with each other.
    """
    from app.api.repositories import index_repository_folder

    counter = {"n": next(_REPO_SEQUENCE)}

    def _build(files: dict, repo_id: str = None):
        counter["n"] = next(_REPO_SEQUENCE)
        root = tmp_path / f"repo{counter['n']}"
        for rel, content in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        rid = repo_id or f"tmp_repo_{counter['n']}"
        result = index_repository_folder(str(root), rid, rid)
        return str(root), rid, result["all_chunks"]

    return _build


@pytest.fixture
def api_client():
    """A TestClient with indexing bootstrap disabled, so tests control what is indexed."""
    from fastapi.testclient import TestClient
    import app.main as main_module

    original = settings.SAMPLE_REPO_DIR
    settings.SAMPLE_REPO_DIR = "/nonexistent-so-bootstrap-is-a-noop"
    try:
        with TestClient(main_module.app) as client:
            yield client
    finally:
        settings.SAMPLE_REPO_DIR = original


# ---------------------------------------------------------------- graph isolation
#
# Neo4j Community supports exactly one user database, so tests cannot be given their own the way
# Mongo and Chroma are. The graph therefore has to be cleaned by repository id instead.
#
# That cleanup used to be a hand-written allowlist ('tmp_repo_*', 'test_sample', 'rx',
# 'stream_repo'). Every new test had to remember to extend it, and the first two that forgot
# ("cite" and "resume") leaked their nodes into the developer's real graph, alongside
# 'test_indexing_repo' from an older run. A list that must be maintained by hand to preserve
# isolation is not isolation.
#
# So the ids are recorded as they are written, and exactly those are deleted. Nothing to keep in
# step. The ledger file carries ids across a crashed run, where teardown never executes.

_GRAPH_REPO_IDS = set()
_GRAPH_ID_LEDGER = os.path.join(tempfile.gettempdir(), "ci_test_graph_repo_ids.txt")


def _record_graph_repo_id(repository_id: str) -> None:
    if repository_id in _GRAPH_REPO_IDS:
        return
    _GRAPH_REPO_IDS.add(repository_id)
    try:
        with open(_GRAPH_ID_LEDGER, "a", encoding="utf-8") as handle:
            handle.write(f"{repository_id}\n")
    except Exception:
        pass


def _ledger_ids() -> set:
    try:
        with open(_GRAPH_ID_LEDGER, "r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except FileNotFoundError:
        return set()
    except Exception:
        return set()


def _real_repository_ids() -> set:
    """
    Repository ids recorded in the developer's own (non-test) database.

    Neo4j Community has a single database, so the test graph and the real graph are the same
    graph. The recorder captures every id written while tests run — including ids the
    *application* writes, not just fixtures: the startup bootstrap indexes the sample repository
    under the fixed id 'sample_ecommerce_repo', which is exactly what a developer's own copy is
    called. Purging that id deleted 26 real nodes once. Nothing whose id belongs to a real
    repository record is ever purged now, whatever the recorder saw.
    """
    try:
        import pymongo
        client = pymongo.MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=1500)
        real_db = os.getenv("MONGODB_DB_NAME_REAL", "Codebase_Intelligence")
        return {
            doc["repository_id"]
            for doc in client[real_db]["repositories"].find({}, {"repository_id": 1})
            if doc.get("repository_id")
        }
    except Exception:
        # Unreachable Mongo means we cannot prove an id is safe, so protect the known collision.
        return {"sample_ecommerce_repo"}


def _purge_graph(repo_ids: set) -> None:
    """
    Removes a set of repositories from whichever graph backend is live.

    The in-memory fallback needs this as much as Neo4j does. It survives for the whole pytest
    process, so without a purge one test's symbols stay visible to the next — the same
    cross-contamination the Neo4j path was leaking, just bounded by the process rather than
    by the developer's patience.
    """
    repo_ids = set(repo_ids) - _real_repository_ids()
    if not repo_ids:
        return
    try:
        from app.graph.neo4j_client import neo4j_client

        if neo4j_client.is_fallback:
            graph = neo4j_client.fallback.g
            doomed = [n for n in graph.nodes if neo4j_client.fallback._repo(n) in repo_ids]
            graph.remove_nodes_from(doomed)
            return

        with neo4j_client.driver.session() as session:
            session.run(
                "MATCH (x) WHERE x.repo_id IN $ids DETACH DELETE x",
                ids=sorted(repo_ids),
            )
    except Exception:
        pass


def _install_graph_recorder() -> None:
    """
    Wraps the single write path so every repository id reaching the graph is recorded.

    Installed at import time rather than in a fixture, so it is in place before any session-scoped
    fixture (`sample_chunks`) can index anything.
    """
    try:
        from app.graph.neo4j_client import neo4j_client
    except Exception:
        return

    original = neo4j_client.build_repository_graph
    if getattr(original, "_records_test_ids", False):
        return

    def recording(repository_id, chunks):
        _record_graph_repo_id(repository_id)
        return original(repository_id, chunks)

    recording._records_test_ids = True
    neo4j_client.build_repository_graph = recording


_install_graph_recorder()


@pytest.fixture(scope="session", autouse=True)
def _clean_test_databases():
    """Drops the test database, and the graph nodes this session (or a crashed one) created."""
    def purge_mongo():
        try:
            import pymongo
            client = pymongo.MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=1500)
            client.drop_database(settings.MONGODB_DB_NAME)
        except Exception:
            pass

    purge_mongo()
    # Anything a previous run recorded but never got to delete.
    _purge_graph(_ledger_ids())

    yield

    purge_mongo()
    _purge_graph(_GRAPH_REPO_IDS | _ledger_ids())
    try:
        os.remove(_GRAPH_ID_LEDGER)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def _remove_test_root():
    """
    Deletes this session's isolated DATA_DIR and CHROMADB_DIR when it ends.

    The isolation was correct; it simply never cleaned up after itself. A session writes real
    ChromaDB collections and index files for every repository it creates, so each run left tens
    of megabytes behind — invisible, in a system temp directory, and cumulative.
    """
    yield
    import shutil
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _close_stores_at_session_end():
    """
    Closes the vector client when the session ends.

    ChromaDB can abort the process during interpreter teardown (SIGABRT,
    `recursive_mutex lock failed`) after tests have already reported success — see F-36. Closing
    deterministically is correct hygiene regardless; it has not been shown to prevent the abort.
    """
    yield
    try:
        from app.vector.chromadb_client import chroma_store
        chroma_store.close()
    except Exception:
        pass
