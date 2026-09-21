"""
The graph view tells the truth about itself (G-08).

A force-directed layout cannot usefully render a large repository, so the cap cannot simply be
removed. What was wrong was being silent about it — a 947-node repository rendered 200 nodes with
nothing in the response to say so — and that the 200 were an arbitrary slice, because `LIMIT` was
applied with no `ORDER BY`.
"""
import pytest

from app.graph.neo4j_client import neo4j_client


@pytest.fixture
def big_repo(temp_repo):
    """More nodes than a small cap, with one obvious hub."""
    files = {"core/hub.py": "def hub():\n    return 1\n"}
    for i in range(30):
        files[f"mod/leaf_{i}.py"] = (
            "from core.hub import hub\n\n"
            f"def leaf_{i}():\n    return hub()\n"
        )
    return temp_repo(files)


# ------------------------------------------------------------------ honesty

def test_the_response_states_the_totals(big_repo):
    _, repo_id, _ = big_repo
    data = neo4j_client.get_visual_graph_data(repo_id)

    assert data["total_nodes"] > 0
    assert data["shown_nodes"] == len(data["nodes"])
    assert data["shown_edges"] == len(data["edges"])
    assert data["total_nodes"] >= data["shown_nodes"]


def test_truncation_is_reported_when_it_happens(big_repo):
    """The whole of the defect: showing a fraction while implying it is the whole."""
    _, repo_id, _ = big_repo
    data = neo4j_client.get_visual_graph_data(repo_id, limit=5)

    assert data["truncated"] is True
    assert data["shown_nodes"] == 5
    assert data["total_nodes"] > 5, "fixture no longer exceeds the cap"
    assert data["node_limit"] == 5


def test_truncated_is_false_when_everything_fits(big_repo):
    """An honest flag has to be honest in both directions, or the UI cries wolf."""
    _, repo_id, _ = big_repo
    data = neo4j_client.get_visual_graph_data(repo_id, limit=5000)

    assert data["truncated"] is False
    assert data["shown_nodes"] == data["total_nodes"]


# ------------------------------------------------------------------ a meaningful slice

def test_the_surviving_nodes_are_the_most_connected(big_repo):
    """
    `LIMIT` with no `ORDER BY` returns whatever the store hands back first. The hub of a
    dependency graph is the one node you cannot afford to have dropped at random.
    """
    _, repo_id, _ = big_repo
    data = neo4j_client.get_visual_graph_data(repo_id, limit=3)

    names = {n["name"] for n in data["nodes"]}
    assert "hub" in names, f"the most-connected node was dropped; kept {names}"


def test_the_slice_is_ordered_by_degree(big_repo):
    _, repo_id, _ = big_repo
    data = neo4j_client.get_visual_graph_data(repo_id, limit=10)

    degrees = [n.get("degree", 0) for n in data["nodes"]]
    assert degrees == sorted(degrees, reverse=True), f"not ordered by connectedness: {degrees}"


def test_the_same_repository_yields_the_same_picture_twice(big_repo):
    """Ties broken by name, so the view does not reshuffle between reloads."""
    _, repo_id, _ = big_repo
    first = neo4j_client.get_visual_graph_data(repo_id, limit=8)
    second = neo4j_client.get_visual_graph_data(repo_id, limit=8)

    assert [n["id"] for n in first["nodes"]] == [n["id"] for n in second["nodes"]]


# ------------------------------------------------------------------ edges stay resolvable

def test_every_edge_refers_to_a_node_that_was_returned(big_repo):
    """
    Node and edge limits used to be applied independently, so edges were fetched and then thrown
    away for referencing nodes beyond the node cap. Edges are now selected from the node set.
    """
    _, repo_id, _ = big_repo
    data = neo4j_client.get_visual_graph_data(repo_id, limit=6)

    ids = {n["id"] for n in data["nodes"]}
    dangling = [e for e in data["edges"] if e["source"] not in ids or e["target"] not in ids]
    assert dangling == [], f"edges referencing absent nodes: {dangling[:3]}"


def test_a_truncated_view_still_contains_the_hub_edges(big_repo):
    """Keeping the hub is only useful if its edges come with it."""
    _, repo_id, _ = big_repo
    data = neo4j_client.get_visual_graph_data(repo_id, limit=10)

    hub_ids = {n["id"] for n in data["nodes"] if n["name"] == "hub"}
    touching = [e for e in data["edges"] if e["source"] in hub_ids or e["target"] in hub_ids]
    assert touching, "the hub was returned with no edges at all"


# ------------------------------------------------------------------ through the API

def test_the_endpoint_surfaces_the_counts(api_client, big_repo):
    _, repo_id, _ = big_repo
    body = api_client.get(f"/api/graph/{repo_id}", params={"limit": 4}).json()

    assert body["truncated"] is True
    assert body["shown_nodes"] == 4
    assert body["total_nodes"] > 4


def test_the_endpoint_rejects_an_absurd_limit(api_client, big_repo):
    _, repo_id, _ = big_repo
    assert api_client.get(f"/api/graph/{repo_id}", params={"limit": 0}).status_code == 422
    assert api_client.get(f"/api/graph/{repo_id}", params={"limit": 99999}).status_code == 422


def test_an_empty_repository_reports_zero_rather_than_truncated(api_client):
    body = api_client.get("/api/graph/no_such_repo").json()
    assert body["total_nodes"] == 0
    assert body["truncated"] is False
    assert body["nodes"] == []
