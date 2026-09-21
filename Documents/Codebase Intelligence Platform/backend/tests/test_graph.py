"""
Call graph construction and traversal. Covers F-08, F-09, F-26, F-35, F-43.

These assert through the public client rather than the in-memory graph, so they exercise
whichever backend is configured. That matters: the Cypher path had never been run when it was
written, and running it against a live server immediately exposed missing repository scoping
(F-43) that the in-memory fallback could not reveal.
"""
import pytest

from app.graph.neo4j_client import neo4j_client, Neo4jGraphClient

REPO = "test_sample"
HTTP_METHODS = ("POST ", "GET ", "PUT ", "DELETE ", "PATCH ")


def _is_route(symbol: str) -> bool:
    return str(symbol).startswith(HTTP_METHODS)


def _node_names(repository_id):
    return {n["id"] for n in neo4j_client.get_visual_graph_data(repository_id)["nodes"]}


# --------------------------------------------------------------------- caller purity (F-08)

def test_callers_are_invocations_not_containment(sample_chunks):
    """A CONTAINS edge (file -> symbol) must never be reported as a caller."""
    callers = neo4j_client.get_callers("calculate_discount", REPO)
    assert callers, "expected real callers for calculate_discount"

    for caller in callers:
        assert caller["relationship"] in ("CALLS", "ROUTES_TO")
        symbol = str(caller["caller_symbol"])
        if not _is_route(symbol):
            assert not symbol.endswith((".py", ".js", ".jsx", ".sql", ".md")), \
                f"{symbol} is a file path, not a caller"


def test_caller_set_is_correct(sample_chunks):
    names = {c["caller_symbol"] for c in neo4j_client.get_callers("calculate_discount", REPO)}
    assert "process_checkout" in names
    assert any(n.startswith("test_") for n in names), "covering tests should appear as callers"


def test_short_symbols_do_not_match_everything(sample_chunks):
    """Bare substring matching made `get` match every node containing those letters."""
    assert len(neo4j_client.get_callers("get", REPO)) < 5


def test_callers_are_deduplicated(sample_chunks):
    """
    A symbol resolving to two nodes returned each caller twice on the Cypher path, where
    DISTINCT dedupes rows rather than callers.
    """
    callers = neo4j_client.get_callers("calculate_discount", REPO)
    names = [c["caller_symbol"] for c in callers]
    assert len(names) == len(set(names)), f"duplicate callers: {names}"


# --------------------------------------------------------------------- repository scoping (F-43)

def test_callers_do_not_leak_between_repositories(temp_repo):
    """
    Two repositories defining the same symbol name must not see each other's callers. The
    Cypher queries had no repo filter, so one project answered with another project's callers.
    """
    _, repo_a, _ = temp_repo({"m.py": "def handler():\n    return helper()\n\ndef helper():\n    return 1\n"})
    _, repo_b, _ = temp_repo({"m.py": "def caller_in_b():\n    return handler()\n\ndef handler():\n    return 2\n"})

    in_a = {c["caller_symbol"] for c in neo4j_client.get_callers("handler", repo_a)}
    in_b = {c["caller_symbol"] for c in neo4j_client.get_callers("handler", repo_b)}

    assert "caller_in_b" in in_b
    assert "caller_in_b" not in in_a, "repository A saw repository B's caller"


def test_forward_dependencies_are_scoped(temp_repo, sample_chunks):
    _, repo_id, _ = temp_repo({"m.py": "def solo_entry():\n    return solo_helper()\n\ndef solo_helper():\n    return 1\n"})
    deps = {d["symbol"] for d in neo4j_client.get_forward_dependencies("solo_entry", 2, repo_id)}
    assert "solo_helper" in deps
    assert not any("discount" in d.lower() for d in deps), "leaked symbols from another repository"


# --------------------------------------------------------------------- aliasing (F-26)

def test_one_node_per_function(sample_chunks):
    """
    Call edges were recorded against the bare callee name while definitions were qualified, so
    each function existed twice with its edges split between the two nodes.
    """
    matching = [n for n in _node_names(REPO) if "calculate_discount" in str(n)]
    assert len(matching) == 1, f"expected one node, found {matching}"


def test_external_calls_do_not_become_nodes(sample_chunks):
    """stdlib/builtin callees are untraversable and were the main source of collisions."""
    builtins = {"get", "round", "print", "len", "max", "min", "str", "int", "append"}
    assert not (builtins & _node_names(REPO))


def test_call_target_resolution_prefers_same_file():
    """Targets carry their defining file: node identity is (repo, file, symbol)."""
    definitions = {"handler": [("A.handler", "a.py"), ("B.handler", "b.py")]}
    assert Neo4jGraphClient._resolve_call_target("handler", "b.py", definitions) == [("B.handler", "b.py")]
    # With no same-file candidate, link to all: a false positive is safer than a missed
    # dependency when answering "what breaks if I change this".
    assert set(Neo4jGraphClient._resolve_call_target("handler", "c.py", definitions)) == \
        {("A.handler", "a.py"), ("B.handler", "b.py")}
    assert Neo4jGraphClient._resolve_call_target("json_dumps", "a.py", definitions) == []


# --------------------------------------------------------------------- integrity (F-09)

def test_graph_has_no_dangling_edges(sample_chunks):
    graph = neo4j_client.get_visual_graph_data(REPO)
    node_ids = {n["id"] for n in graph["nodes"]}
    dangling = [e for e in graph["edges"]
                if e["source"] not in node_ids or e["target"] not in node_ids]
    assert not dangling, f"{len(dangling)} edges reference nodes that were not returned"


def test_every_node_has_a_display_name(sample_chunks):
    graph = neo4j_client.get_visual_graph_data(REPO)
    assert graph["nodes"], "graph should not be empty"
    assert all(n.get("name") for n in graph["nodes"])


def test_repositories_are_isolated_in_the_graph(sample_chunks, temp_repo):
    _, other_id, _ = temp_repo({"solo.py": "def only_here():\n    return 1\n"})
    assert not (_node_names(REPO) & _node_names(other_id))


# --------------------------------------------------------------------- routes (F-35)

@pytest.mark.parametrize("call,expected", [
    ("POST /api/checkout", "process_checkout"),
    ("POST /api/checkout?trace=1", "process_checkout"),
    ("GET /repositories/abc123", "get_repository"),
    ("GET /repositories/*", "get_repository"),          # template literal interpolation
    ("POST http://localhost:8000/api/checkout", "process_checkout"),
    ("GET /api/checkout", None),                         # method mismatch
    ("POST /nope/missing", None),                        # unknown route
])
def test_http_calls_resolve_to_their_handler(call, expected):
    index = Neo4jGraphClient._build_endpoint_index([
        {"entity_type": "endpoint", "symbol": "process_checkout",
         "route": "POST /api/checkout", "file_path": "api/checkout.py"},
        {"entity_type": "endpoint", "symbol": "get_repository",
         "route": "GET /repositories/{id}", "file_path": "api/repos.py"},
    ])
    resolved = Neo4jGraphClient._resolve_http_target(call, index)
    # Resolution returns (symbol, defining_file) so the ROUTES_TO edge targets the right node.
    assert (resolved[0] if resolved else None) == expected


def test_frontend_is_linked_to_the_backend_handler(sample_chunks):
    """The trace must cross the network boundary, not stop at it."""
    callers = neo4j_client.get_callers("process_checkout", REPO)
    routed = [c for c in callers if c["relationship"] == "ROUTES_TO"]
    assert routed, "expected a ROUTES_TO edge into the handler"
    assert any("submitCheckout" in c["caller_symbol"] for c in routed)


def test_frontend_participates_in_the_call_graph(sample_chunks):
    callers = {c["caller_symbol"] for c in neo4j_client.get_callers("submitCheckout", REPO)}
    assert "CheckoutComponent" in callers
