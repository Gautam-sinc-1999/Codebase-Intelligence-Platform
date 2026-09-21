"""
Entity kinds survive the trip through the graph (G-09).

G-09 reported that Neo4j lost the entity-type information the in-memory fallback keeps, making a
function, a class and a method indistinguishable once written. It was recorded as still open on
2026-09-19 on the strength of a probe that queried `n.entity_type` — the property is `n.type`, so
the probe was wrong and the data had been there all along.

These tests exist so the question does not have to be re-litigated by hand: they assert the
property end to end, on both backends, and would fail if a future change dropped it.
"""
import pytest

from app.graph.neo4j_client import neo4j_client


@pytest.fixture
def typed_repo(temp_repo):
    """One of each kind, so a test can tell them apart."""
    return temp_repo({
        "svc.py": (
            "class OrderService:\n"
            "    def create(self, payload):\n"
            "        return validate(payload)\n"
            "\n"
            "def validate(payload):\n"
            "    return True\n"
            "\n"
            '@app.route("/api/orders", methods=["POST"])\n'
            "def create_order():\n"
            "    return OrderService().create({})\n"
        ),
    })


def test_every_kind_is_distinguishable_after_indexing(typed_repo):
    _, repo_id, chunks = typed_repo

    kinds = {c["symbol"]: c["entity_type"] for c in chunks}
    assert kinds["OrderService"] == "class"
    assert kinds["OrderService.create"] == "method"
    assert kinds["validate"] == "function"
    assert kinds["create_order"] == "endpoint"


def test_dependency_results_carry_the_kind(typed_repo):
    """
    A caller needs to know whether it is looking at a class or a function.

    Capitalised, matching `get_visual_graph_data` and the UI's colour map. Neo4j previously
    returned the raw lowercase stored type here while the fallback capitalised, so this is the
    unified contract rather than either backend's old behaviour.
    """
    _, repo_id, _ = typed_repo

    deps = {d["symbol"]: d["label"] for d in
            neo4j_client.get_forward_dependencies("create_order", 2, repo_id)}

    assert deps.get("OrderService") == "Class"
    assert deps.get("OrderService.create") == "Method"
    assert deps.get("validate") == "Function"


def test_the_visual_graph_carries_the_kind(typed_repo):
    """The graph view colours and groups by kind; 'Symbol' for everything renders as a blob."""
    _, repo_id, _ = typed_repo

    labels = {n.get("name"): n.get("label") for n in
              neo4j_client.get_visual_graph_data(repo_id).get("nodes", [])}

    assert labels.get("OrderService") == "Class"
    assert labels.get("OrderService.create") == "Method"
    assert labels.get("validate") == "Function"
    assert labels.get("create_order") == "Endpoint"
    assert any(v == "File" for v in labels.values())


def test_no_symbol_is_left_without_a_kind(typed_repo):
    """
    Call targets are MERGEd by name before their own chunk is written. A target that never gets a
    chunk of its own would keep whatever it was created with — so check none are left untyped.
    """
    _, repo_id, _ = typed_repo

    nodes = neo4j_client.get_visual_graph_data(repo_id).get("nodes", [])
    untyped = [n.get("name") for n in nodes if not n.get("label") or n.get("label") == "Symbol"]
    assert untyped == [], f"nodes with no distinguishable kind: {untyped}"


def test_an_unresolvable_external_call_does_not_create_a_bare_node(temp_repo):
    """`requests.get(...)` is not part of this repository and must not appear as a typeless node."""
    _, repo_id, _ = temp_repo({
        "a.py": "import requests\n\ndef fetch_it():\n    return requests.get('http://x')\n",
    })

    names = {n.get("name") for n in neo4j_client.get_visual_graph_data(repo_id).get("nodes", [])}
    assert "requests.get" not in names and "get" not in names


# ------------------------------------------------------------------ the backends must agree

def test_dependency_kinds_use_the_same_casing_on_both_backends(typed_repo):
    """
    Neo4j returned the stored lowercase type while the fallback returned a capitalised label, so
    the same question got two different answers depending on which backend happened to be up.

    Capitalised is the target because that is what `get_visual_graph_data` already returns on
    both, and what the UI's colour map keys on.
    """
    _, repo_id, _ = typed_repo

    labels = {d["label"] for d in
              neo4j_client.get_forward_dependencies("create_order", 2, repo_id)}
    assert labels, "no dependencies resolved"
    assert all(label[0].isupper() for label in labels), f"uncapitalised labels: {labels}"


def test_a_symbol_is_not_listed_among_its_own_dependencies(typed_repo):
    """
    The in-memory backend returned the starting symbol at depth 0 and Neo4j did not — a quiet
    disagreement, in the direction that makes a fallback untrustworthy exactly when relied upon.
    """
    _, repo_id, _ = typed_repo

    deps = neo4j_client.get_forward_dependencies("create_order", 2, repo_id)
    assert "create_order" not in {d["symbol"] for d in deps}
    assert all(d["depth"] >= 1 for d in deps)


def test_max_depth_is_honoured_rather_than_hardcoded(temp_repo):
    """
    The Neo4j query hardcoded `*1..2`, so the parameter was accepted and ignored. A caller asking
    for one hop got two, which is not a smaller answer — it is a different one.
    """
    _, repo_id, _ = temp_repo({
        "chain.py": (
            "def level_one():\n    return level_two()\n\n"
            "def level_two():\n    return level_three()\n\n"
            "def level_three():\n    return 3\n"
        ),
    })

    one_hop = {d["symbol"] for d in neo4j_client.get_forward_dependencies("level_one", 1, repo_id)}
    two_hops = {d["symbol"] for d in neo4j_client.get_forward_dependencies("level_one", 2, repo_id)}

    assert one_hop == {"level_two"}, f"depth 1 returned {one_hop}"
    assert two_hops == {"level_two", "level_three"}, f"depth 2 returned {two_hops}"
