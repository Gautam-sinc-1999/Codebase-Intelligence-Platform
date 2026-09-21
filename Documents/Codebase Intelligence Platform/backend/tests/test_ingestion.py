"""Discovery, parsing and chunking. Covers F-24, F-25, F-26, F-27, F-31."""
import os

from app.ingestion.discovery import FileDiscovery
from app.ingestion.ast_parser import MultiLanguageASTParser as Parser
from app.ingestion.treesitter_parser import TreeSitterParser


# --------------------------------------------------------------------- secrets (F-31)

SECRET_NAMES = [".env", ".env.local", ".env.production", "server.pem", "id_rsa",
                "credentials.json", "service-account-key.json", ".npmrc"]
TEMPLATE_NAMES = [".env.example", ".env.sample", ".env.template", ".env.dist"]


def test_secret_files_are_never_indexed():
    for name in SECRET_NAMES:
        assert FileDiscovery.is_secret_file(name), f"{name} must be treated as a secret"
        assert not FileDiscovery.should_index(name)


def test_env_templates_are_indexed():
    """Committed templates carry key names without values and are a legitimate question."""
    for name in TEMPLATE_NAMES:
        assert not FileDiscovery.is_secret_file(name), f"{name} is a template, not a secret"
        assert FileDiscovery.should_index(name)


def test_secrets_are_excluded_from_a_real_tree(tmp_path):
    (tmp_path / "app.py").write_text("def f(): pass\n")
    (tmp_path / ".env").write_text("GROQ_API_KEY=sk-live-SECRET\n")
    (tmp_path / "key.pem").write_text("-----BEGIN PRIVATE KEY-----\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("name: CI\n")

    files = FileDiscovery.discover_repository(str(tmp_path), "t")
    paths = {f["relative_path"] for f in files}
    blob = " ".join(f["content"] for f in files)

    assert "sk-live-SECRET" not in blob
    assert "BEGIN PRIVATE KEY" not in blob
    assert not any(p.endswith((".env", "key.pem")) for p in paths)
    # F-31: useful dotfiles and dot-directories must still be reachable.
    assert any("ci.yml" in p for p in paths), "CI config should be indexable"


# --------------------------------------------------------------------- limits (F-24)

def test_oversized_files_are_skipped(tmp_path):
    from app.core.config import settings
    (tmp_path / "small.py").write_text("def f(): pass\n")
    (tmp_path / "huge.js").write_text("x" * (settings.MAX_INDEXED_FILE_BYTES + 1024))

    paths = {f["relative_path"] for f in FileDiscovery.discover_repository(str(tmp_path), "t")}
    assert "small.py" in paths
    assert "huge.js" not in paths


def test_ignored_directories_are_skipped(tmp_path):
    for junk in ["node_modules/pkg/i.js", ".venv/lib/x.py", "__pycache__/m.pyc", "dist/b.js"]:
        target = tmp_path / junk
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n")
    (tmp_path / "real.py").write_text("def real(): pass\n")

    paths = {f["relative_path"] for f in FileDiscovery.discover_repository(str(tmp_path), "t")}
    assert paths == {"real.py"}


def test_discovery_streams_one_file_at_a_time(tmp_path):
    (tmp_path / "a.py").write_text("def a(): pass\n")
    (tmp_path / "b.py").write_text("def b(): pass\n")
    iterator = FileDiscovery.iter_repository(str(tmp_path), "t")
    assert hasattr(iterator, "__next__")
    assert isinstance(next(iterator), dict)


# --------------------------------------------------------------------- parsing (F-25, F-27)

JSX_SOURCE = """import React from 'react';
import { postCheckout } from '../services/checkoutApi';

export function Checkout({ items }) {
  const submit = async () => {
    const res = await postCheckout(items);
    return res;
  };
  return <button onClick={submit}>Pay</button>;
}

class CartStore {
  addItem(item) {
    return recalculate(item);
  }
}

const helper = (x) => compute(x) + 1;
"""

JAVA_SOURCE = """package com.shop;

public class OrderService {
    public Order place(Cart cart) {
        return gateway.charge(calculateTotal(cart));
    }

    private double calculateTotal(Cart cart) {
        return cart.sum();
    }
}
"""


def _entities(content, path, language):
    return Parser.parse_file({
        "repository_id": "t", "relative_path": path,
        "language": language, "content": content,
    })


def test_treesitter_is_available():
    """The regex fallback invents line ranges; the real parser must be in use."""
    assert TreeSitterParser.available()


def _line_of(source, needle):
    """1-based line number of the first line containing `needle`."""
    for index, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} not found in fixture")


def test_js_line_ranges_are_real_not_fabricated():
    """
    Expectations are derived from the fixture rather than hand-counted: the regex parser these
    replaced invented extents as `start + 30`, and a hardcoded expectation can be just as wrong
    as the code it checks.
    """
    entities = {e.symbol: e for e in _entities(JSX_SOURCE, "Checkout.jsx", "react_jsx")}
    total_lines = len(JSX_SOURCE.splitlines())

    assert entities["Checkout"].start_line == _line_of(JSX_SOURCE, "export function Checkout")
    assert entities["CartStore"].start_line == _line_of(JSX_SOURCE, "class CartStore")
    assert entities["CartStore.addItem"].start_line == _line_of(JSX_SOURCE, "addItem(item)")
    assert entities["helper"].start_line == _line_of(JSX_SOURCE, "const helper")
    assert entities["helper"].end_line == entities["helper"].start_line, "one-line arrow function"

    # A definition must enclose its own opening line and stop inside the file.
    for entity in entities.values():
        assert entity.start_line <= entity.end_line
        assert entity.end_line <= total_lines, f"{entity.symbol} claims a line past EOF"

    # The class must fully contain its method — the regex parser could not know where either ended.
    cls_entity, method = entities["CartStore"], entities["CartStore.addItem"]
    assert cls_entity.start_line < method.start_line
    assert method.end_line <= cls_entity.end_line


def test_js_calls_are_extracted(): # F-27: regex extracted only fetch() URLs
    entities = {e.symbol: e for e in _entities(JSX_SOURCE, "Checkout.jsx", "react_jsx")}
    assert "postCheckout" in entities["Checkout"].calls
    assert "recalculate" in entities["CartStore.addItem"].calls
    assert "compute" in entities["helper"].calls


def test_class_does_not_duplicate_method_call_edges():
    entities = {e.symbol: e for e in _entities(JSX_SOURCE, "Checkout.jsx", "react_jsx")}
    assert not entities["CartStore"].calls, "class-level calls duplicate each method's edges"


def test_java_is_parsed_with_calls():  # F-27: regex extracted none at all
    entities = {e.symbol: e for e in _entities(JAVA_SOURCE, "OrderService.java", "java")}
    assert entities["OrderService"].start_line == _line_of(JAVA_SOURCE, "public class OrderService")
    assert "calculateTotal" in entities["OrderService.place"].calls
    assert all(e.end_line <= len(JAVA_SOURCE.splitlines()) for e in entities.values())


# --------------------------------------------------------------------- endpoints (F-11, F-29)

def test_router_prefix_is_resolved_and_empty_routes_normalised():
    cases = [
        ('router = APIRouter(prefix="/checkout")\n@router.post("")\ndef handler(): pass', "POST /checkout"),
        ('router = APIRouter()\n@router.post("")\ndef handler(): pass', "POST /"),
        ('router = APIRouter(prefix="/repos")\n@router.get("/{id}")\ndef handler(): pass', "GET /repos/{id}"),
        ('router = APIRouter(prefix="/a/")\n@router.delete("/b/")\ndef handler(): pass', "DELETE /a/b"),
    ]
    for source, expected in cases:
        endpoints = [e for e in _entities(source, "api.py", "python") if e.entity_type == "endpoint"]
        assert len(endpoints) == 1
        assert endpoints[0].route == expected
        assert not endpoints[0].route.endswith(" "), "malformed 'POST ' style symbol"


def test_handler_is_one_entity_carrying_its_route():
    """F-29: a route and its function were two entities over identical lines."""
    source = 'router = APIRouter(prefix="/checkout")\n@router.post("")\ndef process_checkout():\n    return 1\n'
    entities = _entities(source, "api.py", "python")
    handlers = [e for e in entities if e.start_line == 3]
    assert len(handlers) == 1
    assert handlers[0].symbol == "process_checkout"
    assert handlers[0].entity_type == "endpoint"
    assert handlers[0].route == "POST /checkout"


def test_chunker_preserves_line_numbers_and_rebuildable_content(sample_chunks):
    from app.ingestion.chunker import HierarchicalCodeChunker
    for chunk in sample_chunks:
        assert chunk["start_line"] >= 1
        assert chunk["end_line"] >= chunk["start_line"]
        rebuilt = HierarchicalCodeChunker.format_chunk_content(chunk)
        assert chunk["symbol"] in rebuilt


# ============================================================ language-idiom coverage (G-15..G-20)
#
# "Supported" is not the same as "correct". These lock in idioms that were silently missed:
# each was found by probing the six documented languages against ordinary code.

def test_python_indexes_nested_functions_and_classes():
    """Closures and `class Meta` / `class Config` are ordinary Python and were skipped."""
    source = (
        "class Base:\n"
        "    class Inner:\n"
        "        def inner_method(self): pass\n"
        "    def value(self): return 1\n"
        "\n"
        "def outer():\n"
        "    def nested(): return 1\n"
        "    return nested()\n"
    )
    entities = {e.symbol: e.entity_type for e in _entities(source, "m.py", "python")}
    assert entities["Base.Inner"] == "class"
    assert entities["Base.Inner.inner_method"] == "method"
    assert entities["Base.value"] == "method"
    # A closure is still a function, not a method — "method" means "declared in a class".
    assert entities["outer.nested"] == "function"


TS_CONTAINERS = """export interface User { id: string; }
export abstract class BaseRepo<T> {
  abstract find(id: string): Promise<T>;
  protected helper(): number { return 1; }
}
export namespace Utils { export function fmt(): string { return ""; } }
"""


def test_typescript_abstract_classes_and_namespaces_are_containers():
    """
    `abstract_class_declaration` was not a recognised container, so its methods were emitted
    unqualified — losing the owner and colliding with every other `helper` in the repository.
    """
    entities = {e.symbol for e in _entities(TS_CONTAINERS, "repo.ts", "typescript")}
    assert "BaseRepo" in entities
    assert "BaseRepo.find" in entities and "BaseRepo.helper" in entities
    assert "helper" not in entities, "method escaped its class"
    assert "Utils" in entities and "Utils.fmt" in entities


def test_java_inner_classes_keep_their_owner():
    source = (
        "public class Service {\n"
        "    public int find() { return 1; }\n"
        "    static class Helper { void assist() {} }\n"
        "}\n"
    )
    entities = {e.symbol for e in _entities(source, "S.java", "java")}
    assert "Service.Helper" in entities
    assert "Service.Helper.assist" in entities
    assert "Helper" not in entities, "inner class lost its outer prefix"


JSX_COMPOSITION = """import { memo, forwardRef } from 'react';
import { Card } from './Card';
export const Memoed = memo(function Inner() { return <p/>; });
export const Ref = forwardRef((props, ref) => <input/>);
export const Plain = function Named() { return 1; };
export function Page() {
  const d = loadData();
  return (<Layout><Header title="x"/><Card data={d}/><div className="plain"/><Modal.Header/></Layout>);
}
"""


def test_jsx_rendering_a_component_is_a_dependency():
    """
    `<Card/>` is how one React component depends on another, but JSX elements are not call
    expressions — so component composition, the actual structure of a React app, was absent
    from the call graph entirely.
    """
    entities = {e.symbol: e for e in _entities(JSX_COMPOSITION, "Page.jsx", "react_jsx")}
    page_calls = set(entities["Page"].calls)

    assert {"Card", "Header", "Layout"} <= page_calls, "rendered components must be dependencies"
    assert "Header" in page_calls, "member components resolve to their final property"
    assert "loadData" in page_calls, "ordinary calls still tracked"
    assert "div" not in page_calls, "plain HTML tags are not dependencies"


def test_wrapped_components_keep_their_exported_binding():
    """`<Memoed/>` elsewhere must resolve here; the inner name is only visible inside."""
    entities = {e.symbol for e in _entities(JSX_COMPOSITION, "Page.jsx", "react_jsx")}
    assert {"Memoed", "Ref", "Plain", "Page"} <= entities
    assert "Inner" not in entities and "Named" not in entities


def test_cross_file_dependencies_resolve_in_every_supported_language(temp_repo):
    """The end-to-end contract: "what depends on X" must work for all six languages."""
    from app.graph.neo4j_client import neo4j_client

    cases = {
        "python": ({"core.py": "def calculate_total(i):\n    return sum(i)\n",
                    "api.py": "from core import calculate_total\n\ndef handle(r):\n    return calculate_total(r)\n"},
                   "calculate_total", "handle"),
        "javascript": ({"core.js": "export function calcTotal(i) { return i.length; }\n",
                        "api.js": "import { calcTotal } from './core';\nexport function handle(r) { return calcTotal(r); }\n"},
                       "calcTotal", "handle"),
        "typescript": ({"core.ts": "export function calcTotal(i: number[]): number { return 1; }\n",
                        "api.ts": "import { calcTotal } from './core';\nexport function handle(r: any) { return calcTotal(r); }\n"},
                       "calcTotal", "handle"),
        "jsx": ({"Card.jsx": "export function Card(){ return <div/>; }\n",
                 "Page.jsx": "import { Card } from './Card';\nexport function Page(){ return <Card/>; }\n"},
                "Card", "Page"),
        "java": ({"Core.java": "public class Core { public int calcTotal() { return 1; } }\n",
                  "Api.java": "public class Api { public int handle() { Core c = new Core(); return c.calcTotal(); } }\n"},
                 "calcTotal", "handle"),
    }

    for language, (files, target, expected_caller) in cases.items():
        _, repo_id, _ = temp_repo(files)
        callers = {c["caller_symbol"] for c in neo4j_client.get_callers(target, repo_id)}
        assert any(expected_caller in c for c in callers), \
            f"{language}: expected {expected_caller} among callers of {target}, got {callers}"


def test_same_named_functions_stay_distinct_in_the_graph(temp_repo):
    """Two `helper()` in different files merged into one node, losing one file entirely."""
    from app.graph.neo4j_client import neo4j_client

    _, repo_id, _ = temp_repo({
        "a/util.py": "def helper():\n    return 1\n",
        "b/util.py": "def helper():\n    return 2\n",
        "main.py": "from a.util import helper\n\ndef run():\n    return helper()\n",
    })
    graph = neo4j_client.get_visual_graph_data(repo_id)
    helpers = [n for n in graph["nodes"] if n["name"] == "helper"]

    assert len(helpers) == 2, f"expected both helpers, got {helpers}"
    assert {h["file_path"] for h in helpers} == {"a/util.py", "b/util.py"}

    node_ids = {n["id"] for n in graph["nodes"]}
    assert not [e for e in graph["edges"]
                if e["source"] not in node_ids or e["target"] not in node_ids]


def test_graph_node_labels_are_entity_kinds(sample_chunks):
    """The UI colours by label; Neo4j returned a generic :Symbol for everything."""
    from app.graph.neo4j_client import neo4j_client
    labels = {n["label"] for n in neo4j_client.get_visual_graph_data("test_sample")["nodes"]}
    assert "File" in labels
    assert labels & {"Function", "Method", "Class", "Component", "Endpoint", "Table"}, \
        f"expected entity kinds, got {labels}"
