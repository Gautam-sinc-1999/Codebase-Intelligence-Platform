"""
Guards the test harness itself.

Mongo and Chroma get their own database and directory. Neo4j Community allows only one user
database, so the graph is instead cleaned by repository id — and the previous mechanism, a
hand-written allowlist, silently failed the moment a test used an id nobody remembered to add.
Two did, and their nodes ended up in the developer's real graph.

These tests assert the replacement actually tracks what it deletes.
"""
import conftest


def test_indexing_a_repository_records_its_id_for_cleanup(temp_repo):
    """The recording wrapper is the only thing standing between tests and the real graph."""
    _, repo_id, _ = temp_repo({"alpha.py": "def alpha():\n    return 1\n"})
    assert repo_id in conftest._GRAPH_REPO_IDS


def test_the_ledger_persists_ids_so_a_crashed_run_is_still_cleaned(temp_repo):
    """Teardown never runs if the session dies; the next run reads the ledger and finishes the job."""
    _, repo_id, _ = temp_repo({"beta.py": "def beta():\n    return 2\n"})
    assert repo_id in conftest._ledger_ids()


def test_purge_deletes_exactly_the_recorded_repository(temp_repo):
    from app.graph.neo4j_client import neo4j_client

    _, doomed, _ = temp_repo({"gamma.py": "def gamma():\n    return 3\n"})
    _, kept, _ = temp_repo({"delta.py": "def delta():\n    return 4\n"})

    def node_count(repo_id):
        """Counts straight from the live backend, so the assertion tracks what purge changes."""
        if neo4j_client.is_fallback:
            fallback = neo4j_client.fallback
            return sum(1 for n in fallback.g.nodes if fallback._repo(n) == repo_id)
        with neo4j_client.driver.session() as session:
            return session.run(
                "MATCH (x) WHERE x.repo_id = $id RETURN count(x) AS n", id=repo_id
            ).single()["n"]

    assert node_count(doomed) > 0 and node_count(kept) > 0

    conftest._purge_graph({doomed})

    assert node_count(doomed) == 0, "purge did not remove the repository it was given"
    assert node_count(kept) > 0, "purge removed a repository it was not given"


def test_recorder_is_installed_once_even_if_conftest_is_reimported():
    """Double-wrapping would keep working but makes the call stack harder to read; guard it."""
    from app.graph.neo4j_client import neo4j_client

    before = neo4j_client.build_repository_graph
    conftest._install_graph_recorder()
    assert neo4j_client.build_repository_graph is before
