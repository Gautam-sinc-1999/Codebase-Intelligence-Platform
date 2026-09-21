"""
Flask route detection (G-04).

The parser handled `@router.post("/x")`, which FastAPI and Flask 2.0's method shorthand share.
Classic `@app.route(...)` — still the dominant Flask idiom — produced no endpoint at all, so a
Flask application got no ROUTES_TO edges and the cross-stack chain that works for FastAPI yielded
nothing.
"""
import pytest

from app.ingestion.ast_parser import MultiLanguageASTParser as Parser
from app.graph.neo4j_client import neo4j_client


def _parse(source: str, path: str = "app.py"):
    return Parser.parse_file({
        "relative_path": path, "content": source, "language": "python",
        "repository_id": "r", "line_count": source.count("\n") + 1,
    })


def _routes(source: str, path: str = "app.py"):
    return {e.symbol: e.route for e in _parse(source, path) if e.route}


# ------------------------------------------------------------------ decorator shapes

def test_a_bare_route_defaults_to_get_as_flask_does():
    """The whole of the defect: a bare `@app.route` is a real GET endpoint, and was skipped."""
    routes = _routes('@app.route("/health")\ndef health():\n    return {}\n')
    assert routes == {"health": "GET /health"}


def test_an_explicit_method_is_honoured():
    routes = _routes('@app.route("/checkout", methods=["POST"])\ndef checkout():\n    return {}\n')
    assert routes == {"checkout": "POST /checkout"}


def test_several_methods_become_one_endpoint_not_several():
    """
    Emitting one entity per method would split the call graph across nodes covering identical
    lines — the same mistake that separate route-and-function entities made.
    """
    entities = _parse(
        '@app.route("/items", methods=["GET", "POST", "DELETE"])\ndef items():\n    return []\n')
    endpoints = [e for e in entities if e.route]
    assert len(endpoints) == 1, "one handler must be one node"
    assert endpoints[0].route == "DELETE|GET|POST /items"


def test_methods_are_sorted_so_the_index_is_reproducible():
    a = _routes('@app.route("/x", methods=["POST", "GET"])\ndef h():\n    return {}\n')
    b = _routes('@app.route("/x", methods=["GET", "POST"])\ndef h():\n    return {}\n')
    assert a == b


def test_the_fastapi_and_flask_shorthand_forms_still_work():
    """These already worked; the change must not disturb them."""
    assert _routes('@app.get("/modern")\ndef modern():\n    return {}\n') == {"modern": "GET /modern"}
    assert _routes('@router.post("/x")\ndef x():\n    return {}\n') == {"x": "POST /x"}


def test_a_path_parameter_is_preserved_verbatim():
    routes = _routes('@app.route("/orders/<int:order_id>")\ndef get_order(order_id):\n    return {}\n')
    assert routes == {"get_order": "GET /orders/<int:order_id>"}


# ------------------------------------------------------------------ what must NOT become a route

def test_methods_that_cannot_be_read_statically_do_not_guess_get():
    """
    `methods=ALLOWED` is not knowable at parse time. Defaulting to GET would assert something
    false about the application; leaving it a plain function merely omits an edge.
    """
    entities = _parse('@app.route("/dynamic", methods=ALLOWED)\ndef dynamic():\n    return {}\n')
    dynamic = next(e for e in entities if e.symbol == "dynamic")
    assert dynamic.route is None
    assert dynamic.entity_type == "function"


def test_an_undecorated_function_is_not_an_endpoint():
    entities = _parse("def plain_helper():\n    return 1\n")
    assert all(e.route is None for e in entities)


def test_an_unrelated_decorator_is_not_a_route():
    entities = _parse("@lru_cache(maxsize=1)\ndef cached():\n    return 1\n")
    assert all(e.route is None for e in entities)


# ------------------------------------------------------------------ blueprint prefixes

def test_a_blueprint_url_prefix_is_applied():
    source = (
        'bp = Blueprint("orders", __name__, url_prefix="/api/orders")\n\n'
        '@bp.route("/<int:oid>")\ndef get_order(oid):\n    return {}\n'
    )
    assert _routes(source) == {"get_order": "GET /api/orders/<int:oid>"}


def test_a_blueprint_without_a_prefix_yields_a_bare_path():
    source = 'bp = Blueprint("orders", __name__)\n\n@bp.route("/x")\ndef x():\n    return {}\n'
    assert _routes(source) == {"x": "GET /x"}


def test_the_blueprint_module_name_is_not_mistaken_for_a_prefix():
    """`Blueprint`'s second positional argument is __name__, not a path."""
    assert Parser._detect_router_prefix('bp = Blueprint("orders", __name__)') == ""


def test_apirouter_prefixes_still_take_precedence_for_fastapi():
    source = 'router = APIRouter(prefix="/api/v1")\n\n@router.post("/x")\ndef x():\n    return {}\n'
    assert _routes(source) == {"x": "POST /api/v1/x"}


# ------------------------------------------------------------------ the graph

def test_a_flask_app_gets_the_cross_stack_chain(temp_repo):
    """
    The point of detecting routes at all: a frontend caller resolving through an HTTP boundary
    to the handler, and onward to what the handler calls.
    """
    _, repo_id, _ = temp_repo({
        "api/server.py": (
            "from flask import Flask\napp = Flask(__name__)\n\n"
            '@app.route("/api/checkout", methods=["POST"])\n'
            "def process_checkout():\n    return charge_card()\n\n"
            "def charge_card():\n    return {'ok': True}\n"
        ),
        "ui/Cart.jsx": (
            "export function CartComponent() {\n"
            '  const submit = () => fetch("/api/checkout", { method: "POST" });\n'
            "  return submit;\n}\n"
        ),
    })

    routed = neo4j_client.get_callers("process_checkout", repo_id)
    assert any(c["caller_symbol"] == "CartComponent" and c["relationship"] == "ROUTES_TO"
               for c in routed), "the frontend caller did not reach the Flask handler"

    called = neo4j_client.get_callers("charge_card", repo_id)
    assert any(c["caller_symbol"] == "process_checkout" and c["relationship"] == "CALLS"
               for c in called), "the chain stopped at the handler"


def test_a_multi_method_handler_is_reachable_by_each_verb(temp_repo):
    """One handler, one node, reachable by either verb — which is what a POST caller needs."""
    _, repo_id, _ = temp_repo({
        "api/server.py": (
            '@app.route("/api/items", methods=["GET", "POST"])\n'
            "def items():\n    return []\n"
        ),
        "ui/Get.jsx": 'export function Reader() { return fetch("/api/items"); }\n',
        "ui/Post.jsx": (
            "export function Writer() {\n"
            '  return fetch("/api/items", { method: "POST" });\n}\n'
        ),
    })

    callers = {c["caller_symbol"] for c in neo4j_client.get_callers("items", repo_id)}
    assert "Reader" in callers, "the GET caller did not resolve"
    assert "Writer" in callers, "the POST caller did not resolve"
