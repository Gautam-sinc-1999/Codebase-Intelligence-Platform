from fastapi import APIRouter, Query
from app.graph.neo4j_client import neo4j_client

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/{repository_id}")
async def get_repository_graph(
    repository_id: str,
    limit: int = Query(None, ge=1, le=2000,
                       description="Maximum nodes to return, most connected first."),
):
    """
    A renderable subgraph, plus how much of the whole it represents.

    The response carries `total_nodes`, `shown_nodes` and `truncated` because a force-directed
    layout cannot usefully render a large repository and the cap therefore cannot be removed —
    but a client showing a fifth of a graph must be able to say so rather than presenting it as
    the graph.
    """
    return neo4j_client.get_visual_graph_data(repository_id, limit=limit)
