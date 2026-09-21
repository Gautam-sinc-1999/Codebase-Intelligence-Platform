import os
import logging
from typing import List, Dict, Any, Optional
import networkx as nx
from app.core.config import settings

logger = logging.getLogger("neo4j")

class MemoryGraphFallback:
    """
    In-memory graph standing in for Neo4j when no server is reachable.

    Nodes are keyed by (repository_id, symbol), not by symbol alone. Keying on the bare name made
    the graph global: indexing a second repository that shared any symbol name overwrote the
    first repository's node properties, and indexing the *same* code under a new id wiped the
    earlier repository's graph entirely — 26 nodes to 0 — so its dependency and impact answers
    silently returned nothing. Re-uploading a repository is ordinary use, and this is the default
    backend whenever Neo4j is not running.
    """

    def __init__(self):
        self.g = nx.DiGraph()

    @staticmethod
    def _key(repository_id: Optional[str], symbol: str, file_path: str = "") -> tuple:
        """
        Node identity is (repository, file, symbol).

        Keying on the symbol alone merged two same-named functions into one node, so
        `a/util.py:helper` and `b/util.py:helper` shared callers, a file path and a line range —
        and the last one indexed silently overwrote the other. Same-named helpers are ordinary
        code structure (`utils.py`, `main`, `run`, `handler`), not an edge case.
        """
        return (repository_id or "", str(file_path or ""), str(symbol))

    @staticmethod
    def _symbol(key) -> str:
        return key[2] if isinstance(key, tuple) and len(key) == 3 else (key[1] if isinstance(key, tuple) else str(key))

    @staticmethod
    def _repo(key) -> str:
        return key[0] if isinstance(key, tuple) else ""

    @staticmethod
    def _file(key) -> str:
        return key[1] if isinstance(key, tuple) and len(key) == 3 else ""

    def add_node(self, node_id: str, label: str, properties: Dict[str, Any]):
        repository_id = properties.get("repository_id")
        file_path = properties.get("file_path", "")
        self.g.add_node(self._key(repository_id, node_id, file_path), label=label, **properties)

    def add_relationship(self, source_id: str, target_id: str, rel_type: str,
                         repository_id: Optional[str] = None,
                         source_file: str = "", target_file: str = ""):
        source = self._key(repository_id, source_id, source_file)
        target = self._key(repository_id, target_id, target_file)
        for node in (source, target):
            if not self.g.has_node(node):
                self.g.add_node(node, repository_id=repository_id)
        self.g.add_edge(source, target, type=rel_type)

    def get_forward_dependencies(self, symbol: str, max_depth: int = 2, repository_id: Optional[str] = None) -> List[Dict[str, Any]]:
        start_nodes = self._resolve_target_nodes(symbol, repository_id)
        if not start_nodes:
            return []

        results = []
        visited = set()
        queue = [(start_nodes[0], 0)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            # The starting symbol is not one of its own dependencies. It was being returned at
            # depth 0, so this backend's answer contained one entry the Neo4j backend's did not —
            # the kind of quiet disagreement that makes a fallback untrustworthy precisely when
            # it is being relied on.
            if depth > 0:
                node_data = self.g.nodes[current]
                results.append({
                    "symbol": self._symbol(current),
                    "label": node_data.get("label", "Symbol"),
                    "file_path": node_data.get("file_path", ""),
                    "depth": depth
                })

            for neighbor in self.g.successors(current):
                queue.append((neighbor, depth + 1))
        return results

    # Only these edge types represent an actual invocation. Structural edges — notably
    # CONTAINS (file -> symbol), added for every symbol in build_repository_graph — must never
    # be reported as callers, or the answer claims a *file* calls a function.
    # ROUTES_TO qualifies: a component issuing a request to an endpoint is invoking it, just
    # across the network rather than in-process.
    CALLER_EDGE_TYPES = {"CALLS", "ROUTES_TO"}

    def _resolve_target_nodes(self, symbol: str, repository_id: Optional[str] = None) -> List[str]:
        """
        Resolves a queried symbol to graph nodes, most precise match first.

        A bare substring test matches far too much: querying "get" would match every node whose
        name contains it. Exact match wins, then the dotted tail (so "calculate_discount" finds
        "DiscountService.calculate_discount"), and only then a substring — and never for tokens
        short enough to match indiscriminately.
        """
        sym = symbol.lower().strip()
        if not sym:
            return []

        # Nodes are keyed by (repository_id, symbol); resolution stays inside one repository.
        if repository_id:
            nodes = [n for n in self.g.nodes if self._repo(n) == repository_id]
        else:
            nodes = list(self.g.nodes)
        sym_tail = sym.rsplit(".", 1)[-1]

        # Exact and tail matches are unioned rather than tried in sequence, because a user
        # naturally types the bare name ("calculate_discount") for a node stored qualified
        # ("DiscountService.calculate_discount"). Call-target resolution at graph-build time
        # now guarantees one node per function, so the union can no longer pull in an alias.
        matches = {n for n in nodes if self._symbol(n).lower() == sym}
        matches |= {n for n in nodes if self._symbol(n).lower().rsplit(".", 1)[-1] == sym_tail}
        if matches:
            return sorted(matches)

        if len(sym) > 3:
            return [n for n in nodes if sym in self._symbol(n).lower()]
        return []

    def get_callers(self, symbol: str, repository_id: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        seen = set()

        for target in self._resolve_target_nodes(symbol, repository_id):
            for predecessor in self.g.predecessors(target):
                edge_data = self.g.get_edge_data(predecessor, target) or {}
                relationship = edge_data.get("type", "")

                if relationship not in self.CALLER_EDGE_TYPES:
                    continue

                key = (predecessor, target)
                if key in seen:
                    continue
                seen.add(key)

                node_data = self.g.nodes.get(predecessor, {})
                results.append({
                    "caller_symbol": self._symbol(predecessor),
                    "target_symbol": self._symbol(target),
                    "relationship": relationship,
                    "file_path": node_data.get("file_path", ""),
                    "start_line": node_data.get("start_line", 1)
                })
        return results

    def get_entire_graph_data(self, repo_id: str, limit: int = None) -> Dict[str, Any]:
        """
        The in-memory equivalent of `get_visual_graph_data`, including its truncation reporting.

        Kept deliberately in step with the Neo4j implementation: same cap, same ordering by
        degree, same totals in the response. A view that shows a different subgraph depending on
        which backend is up is a view nobody can reason about.
        """
        candidates = [n for n in self.g.nodes if self._repo(n) == repo_id or not repo_id]
        total_nodes = len(candidates)
        total_edges = sum(
            1 for u, v in self.g.edges()
            if (self._repo(u) == repo_id or not repo_id) and (self._repo(v) == repo_id or not repo_id)
        )

        # Most connected first, so what survives the cap is the part of the graph worth seeing.
        # The name is a tiebreak purely so the same repository yields the same picture twice.
        candidates.sort(key=lambda n: (-self.g.degree(n), self._symbol(n)))
        if limit:
            candidates = candidates[:limit]

        nodes = []
        node_ids = set()
        id_by_key: Dict[tuple, str] = {}

        for n in candidates:
            data = self.g.nodes[n]
            symbol = self._symbol(n)
            # The id must distinguish same-named symbols, while `name` stays readable.
            node_id = f"{self._file(n)}::{symbol}" if self._file(n) else symbol
            nodes.append({
                "id": node_id,
                "label": data.get("label", "Symbol"),
                "name": data.get("name", symbol),
                "file_path": data.get("file_path", ""),
                "degree": self.g.degree(n),
            })
            node_ids.add(n)
            id_by_key[n] = node_id

        # Only emit an edge when BOTH endpoints are in the returned node set. Any node filtered
        # out by repository or by the cap would otherwise leave an edge referencing an id the
        # client cannot resolve.
        edges = [
            {"source": id_by_key[u], "target": id_by_key[v], "type": data.get("type", "DEPENDS_ON")}
            for u, v, data in self.g.edges(data=True)
            if u in node_ids and v in node_ids
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "shown_nodes": len(nodes),
            "shown_edges": len(edges),
            "truncated": total_nodes > len(nodes),
            "node_limit": limit,
        }

class Neo4jGraphClient:
    """
    Neo4j Graph Database Driver with Cypher query execution for Codebase Intelligence.
    Handles graph nodes (File, Class, Function, API, DatabaseTable) and edges (CALLS, IMPORTS, ROUTES_TO, etc.).
    """

    def __init__(self):
        self.driver = None
        self.is_fallback = False
        self.fallback = MemoryGraphFallback()
        self._init_driver()

    def _init_driver(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j at %s", settings.NEO4J_URI)
            self._ensure_schema()
        except Exception as e:
            logger.warning("Neo4j database connection failed (%s). Utilizing Neo4j fallback engine.", e)
            self.is_fallback = True

    def _ensure_schema(self):
        """
        Creates the identity constraints and lookup indexes.

        Without them every MERGE on (:Symbol {name, file_path, repo_id}) is an unindexed label
        scan — one per chunk plus one per call edge — which is why building the graph for an
        857-chunk repository took seconds. The constraint also enforces the identity the code
        assumes rather than leaving it to convention.
        """
        statements = [
            "CREATE CONSTRAINT symbol_identity IF NOT EXISTS "
            "FOR (s:Symbol) REQUIRE (s.name, s.file_path, s.repo_id) IS UNIQUE",
            "CREATE CONSTRAINT file_identity IF NOT EXISTS "
            "FOR (f:File) REQUIRE (f.path, f.repo_id) IS UNIQUE",
            "CREATE INDEX symbol_by_repo IF NOT EXISTS FOR (s:Symbol) ON (s.repo_id)",
            "CREATE INDEX symbol_by_name IF NOT EXISTS FOR (s:Symbol) ON (s.name)",
        ]
        try:
            with self.driver.session() as session:
                for statement in statements:
                    session.run(statement)
            logger.info("Neo4j schema verified (constraints and indexes present).")
        except Exception as e:
            # A read-only or community-edition instance may refuse; indexing still works, slower.
            logger.warning("Could not create Neo4j schema objects: %s", e)

    # Entity kinds that constitute a callable definition within this repository.
    DEFINITION_ENTITY_TYPES = {"function", "method", "class", "component", "endpoint"}

    @classmethod
    def _build_definition_index(cls, chunks: List[Dict[str, Any]]) -> Dict[str, List[tuple]]:
        """Indexes every symbol this repository defines, keyed by its bare (undotted) name."""
        index: Dict[str, List[tuple]] = {}
        for chunk in chunks:
            symbol = chunk.get("symbol", "")
            if not symbol or chunk.get("entity_type") not in cls.DEFINITION_ENTITY_TYPES:
                continue
            tail = symbol.rsplit(".", 1)[-1].lower()
            index.setdefault(tail, []).append((symbol, chunk.get("file_path", "")))
        return index

    @staticmethod
    def _resolve_call_target(call: str, caller_file: str, definition_index: Dict[str, List[tuple]]) -> List[str]:
        """
        Maps a bare callee name onto the qualified symbol(s) this repository actually defines.

        The AST records callees by bare name ('calculate_discount', and for attribute calls just
        the final attribute), so without resolution every '.get()' in the repo collapses into one
        shared node, and a function's definition node ends up distinct from the node its callers
        point at. Resolving here gives one node per logical function.

        Calls with no definition in this repository (stdlib, third-party) return [] and are
        deliberately dropped: they cannot be traversed for impact analysis and their bare names
        are the main source of node collisions.
        """
        if not call:
            return []

        candidates = definition_index.get(call.rsplit(".", 1)[-1].lower(), [])
        if not candidates:
            return []

        # Targets carry their defining file: node identity is (repo, file, symbol), so an edge
        # to "helper" is ambiguous unless it says which helper.
        same_file = [(sym, path) for sym, path in candidates if path == caller_file]
        if same_file:
            return same_file

        # Otherwise link to every candidate. For impact analysis an over-broad edge ("this might
        # be affected") is safer than a missing one ("this is unaffected" when it is not).
        return list(candidates)

    @classmethod
    def _resolve_chunk_calls(cls, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns chunks whose 'calls' are rewritten to resolved, repo-defined qualified symbols."""
        definition_index = cls._build_definition_index(chunks)
        resolved = []

        for chunk in chunks:
            symbol = chunk.get("symbol", "")
            caller_file = chunk.get("file_path", "")

            targets = set()
            for call in chunk.get("calls", []) or []:
                for target_symbol, target_file in cls._resolve_call_target(call, caller_file, definition_index):
                    # Skip self-loops; they add nothing to impact traversal.
                    if not (target_symbol == symbol and target_file == caller_file):
                        targets.add((target_symbol, target_file))

            rewritten = dict(chunk)
            rewritten["calls"] = [{"name": n, "file": f} for n, f in sorted(targets)]
            resolved.append(rewritten)

        return resolved

    @staticmethod
    def _split_route(path: str) -> List[str]:
        """Normalises a route or URL into comparable segments, dropping query and fragment."""
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path.startswith("http"):
            # Strip scheme and host: keep only the path portion.
            parts = path.split("/", 3)
            path = "/" + parts[3] if len(parts) > 3 else "/"
        return [segment for segment in path.strip("/").split("/") if segment]

    @staticmethod
    def _is_route_param(segment: str) -> bool:
        """True for `{id}` (FastAPI), `:id` (Express) and `*` (an interpolated value)."""
        return (
            (segment.startswith("{") and segment.endswith("}"))
            or segment.startswith(":")
            or segment == "*"
        )

    @classmethod
    def _segments_match(cls, route_segments: List[str], url_segments: List[str]) -> bool:
        if len(route_segments) != len(url_segments):
            return False
        return all(
            cls._is_route_param(route) or cls._is_route_param(url) or route.lower() == url.lower()
            for route, url in zip(route_segments, url_segments)
        )

    @classmethod
    def _resolve_http_target(cls, http_call: str, endpoint_index: Dict[str, List[tuple]]) -> Optional[str]:
        """
        Matches an outbound request ("POST /api/checkout") to the endpoint symbol that serves it.

        Exact segment comparison is tried first, then a suffix match, because a prefix applied
        via `include_router(...)` is not resolvable at parse time (see F-11) and leaves the
        declared route shorter than the URL the frontend actually calls.
        """
        method, _, url = http_call.partition(" ")
        if not url:
            return None

        candidates = endpoint_index.get(method.upper(), [])
        if not candidates:
            return None

        url_segments = cls._split_route(url)

        for symbol, route_segments, file_path in candidates:
            if cls._segments_match(route_segments, url_segments):
                return (symbol, file_path)

        for symbol, route_segments, file_path in candidates:
            if route_segments and len(route_segments) <= len(url_segments):
                if cls._segments_match(route_segments, url_segments[-len(route_segments):]):
                    return (symbol, file_path)

        return None

    @classmethod
    def _build_endpoint_index(cls, chunks: List[Dict[str, Any]]) -> Dict[str, List[tuple]]:
        """
        Indexes handlers by HTTP method, keyed on the `route` attribute.

        The target of the resulting ROUTES_TO edge is the handler's own symbol, so a frontend
        caller and in-process callers converge on one node (F-29) instead of the route and the
        function each collecting half the dependents.
        """
        index: Dict[str, List[tuple]] = {}
        for chunk in chunks:
            route_spec = chunk.get("route")
            if not route_spec:
                continue
            method, _, route = route_spec.partition(" ")
            if not method or not route:
                continue
            # A Flask handler declared with methods=["GET", "POST"] arrives as "GET|POST /x" and
            # is registered under each: it is one handler — one graph node — reachable by either
            # verb, which is what a caller issuing a POST needs in order to resolve to it.
            for verb in method.upper().split("|"):
                if not verb:
                    continue
                index.setdefault(verb, []).append(
                    (chunk["symbol"], cls._split_route(route), chunk.get("file_path", ""))
                )
        return index

    def delete_repository(self, repository_id: str) -> int:
        """
        Removes every node belonging to a repository, and the edges attached to them.

        Node identity is (repo_id, file_path, symbol), so `repo_id` alone selects exactly one
        repository's subgraph and can never catch a same-named symbol from another — which is
        the reason that identity was widened in the first place (F-45).

        Returns the number of nodes removed.
        """
        if not repository_id:
            return 0

        if self.is_fallback:
            doomed = [n for n in self.fallback.g.nodes if self.fallback._repo(n) == repository_id]
            self.fallback.g.remove_nodes_from(doomed)
            return len(doomed)

        try:
            with self.driver.session() as session:
                # DETACH so relationships go with the nodes; a left-behind edge would dangle.
                # FOREACH rather than a scoped CALL subquery: both work here, but FOREACH is
                # valid back to Neo4j 4.x and nothing is gained by requiring 5.23+.
                record = session.run(
                    "MATCH (x) WHERE x.repo_id = $repo_id "
                    "WITH collect(x) AS doomed, count(x) AS removed "
                    "FOREACH (d IN doomed | DETACH DELETE d) "
                    "RETURN removed",
                    repo_id=repository_id,
                ).single()
                removed = record["removed"] if record else 0
                logger.info("Deleted %d graph node(s) for repository '%s'.", removed, repository_id)
                return removed
        except Exception as e:
            logger.error("Could not delete graph nodes for '%s': %s", repository_id, e)
            return 0

    def build_repository_graph(self, repository_id: str, chunks: List[Dict[str, Any]]):
        chunks = self._resolve_chunk_calls(chunks)
        endpoint_index = self._build_endpoint_index(chunks)

        if self.is_fallback:
            for chunk in chunks:
                symbol = chunk["symbol"]
                file_path = chunk["file_path"]
                entity_type = chunk["entity_type"]
                
                self.fallback.add_node(
                    symbol,
                    label=entity_type.capitalize(),
                    properties={
                        "repository_id": repository_id,
                        "name": symbol,
                        "file_path": file_path,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"]
                    }
                )

                # File nodes must be created explicitly. Left to add_relationship they would be
                # created bare, without repository_id, and then filtered out of the visual graph
                # while their CONTAINS edges survived — producing edges pointing at nothing.
                self.fallback.add_node(
                    file_path,
                    label="File",
                    properties={
                        "repository_id": repository_id,
                        "name": os.path.basename(file_path) or file_path,
                        "file_path": file_path
                    }
                )
                self.fallback.add_relationship(
                    file_path, symbol, "CONTAINS", repository_id,
                    source_file=file_path, target_file=file_path,
                )

                for call in chunk.get("calls", []):
                    self.fallback.add_relationship(
                        symbol, call["name"], "CALLS", repository_id,
                        source_file=file_path, target_file=call["file"],
                    )

                # ROUTES_TO links a caller to the endpoint that serves its request — the seam
                # between frontend and backend. It previously pointed from a *file* to an
                # endpoint it contained, which merely duplicated CONTAINS and described nothing.
                for http_call in chunk.get("http_calls", []) or []:
                    resolved = self._resolve_http_target(http_call, endpoint_index)
                    if resolved:
                        target_symbol, target_file = resolved
                        self.fallback.add_relationship(
                            symbol, target_symbol, "ROUTES_TO", repository_id,
                            source_file=file_path, target_file=target_file,
                        )
            return

        # ROUTES_TO is built here as well as in the fallback. It was previously only created on
        # the in-memory path, so cross-stack tracing (F-35) silently did nothing against a live
        # Neo4j server — the half of the product that connects a component to the endpoint
        # serving it worked only when the database was unavailable.
        cypher_query = """
        UNWIND $chunks AS chunk
        MERGE (f:File {path: chunk.file_path, repo_id: $repo_id})
        MERGE (s:Symbol {name: chunk.symbol, file_path: chunk.file_path, repo_id: $repo_id})
        SET s.start_line = chunk.start_line,
            s.end_line = chunk.end_line,
            s.type = chunk.entity_type,
            s.route = chunk.route
        MERGE (f)-[:CONTAINS]->(s)
        WITH s, chunk
        FOREACH (call IN chunk.calls |
            MERGE (target:Symbol {name: call.name, file_path: call.file, repo_id: $repo_id})
            MERGE (s)-[:CALLS]->(target)
        )
        FOREACH (handler IN chunk.route_targets |
            MERGE (endpoint:Symbol {name: handler.name, file_path: handler.file, repo_id: $repo_id})
            MERGE (s)-[:ROUTES_TO]->(endpoint)
        )
        """
        # Project only the fields the query reads — the full chunk carries code_snippet and
        # formatted_content, which would ship the entire repository's source to the database.
        payload = [
            {
                "file_path": c.get("file_path", ""),
                "symbol": c.get("symbol", ""),
                "start_line": c.get("start_line", 1),
                "end_line": c.get("end_line", 1),
                "entity_type": c.get("entity_type", ""),
                "route": c.get("route"),
                "calls": c.get("calls", []),
                # Outbound HTTP requests resolved to the handlers that serve them, so the driver
                # path builds the same ROUTES_TO edges as the fallback.
                "route_targets": [
                    {"name": name, "file": file_path}
                    for name, file_path in sorted({
                        resolved
                        for http_call in (c.get("http_calls") or [])
                        for resolved in [self._resolve_http_target(http_call, endpoint_index)]
                        if resolved and resolved[0] != c.get("symbol")
                    })
                ],
            }
            for c in chunks
        ]

        try:
            with self.driver.session() as session:
                session.run(cypher_query, repo_id=repository_id, chunks=payload)
        except Exception as e:
            logger.error("Failed to build Neo4j graph: %s", e)

    def get_callers(self, symbol: str, repository_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.is_fallback:
            return self.fallback.get_callers(symbol, repository_id)

        # Matches the fallback's precedence: exact name, else dotted tail, else substring for
        # tokens long enough to be selective. [:CALLS] already excludes structural edges.
        # Scoped to one repository. Without this the query spans every repository in the
        # database, so "who calls X?" in one project answers with callers from another —
        # a defect invisible against the in-memory fallback and only exposed on a live server.
        cypher = """
        MATCH (caller:Symbol)-[r:CALLS|ROUTES_TO]->(target:Symbol)
        WHERE ($repo_id IS NULL OR (caller.repo_id = $repo_id AND target.repo_id = $repo_id))
        WITH caller, r, target, toLower($symbol) AS sym, toLower(target.name) AS tname
        WHERE tname = sym
           OR split(tname, '.')[-1] = split(sym, '.')[-1]
           OR (size(sym) > 3 AND tname CONTAINS sym)
        RETURN DISTINCT caller.name AS caller_symbol, target.name AS target_symbol,
               type(r) AS relationship, caller.file_path AS file_path, caller.start_line AS start_line
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, symbol=symbol, repo_id=repository_id)
                records = [record.data() for record in result]
        except Exception as e:
            logger.error("Neo4j Cypher caller query failed: %s", e)
            return []

        # DISTINCT dedupes rows, not callers: a symbol resolving to both `calculate_discount`
        # and `DiscountService.calculate_discount` yields each caller once per target.
        deduped, seen = [], set()
        for record in records:
            key = (record.get("caller_symbol"), record.get("relationship"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def get_forward_dependencies(self, symbol: str, max_depth: int = 2, repository_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.is_fallback:
            return self.fallback.get_forward_dependencies(symbol, max_depth, repository_id)

        # The depth bound is interpolated rather than parameterised because Cypher does not
        # accept a parameter inside a variable-length pattern. Clamped to a small integer range
        # first, so the value can only ever be a literal digit — this is not a string built from
        # user input.
        depth = max(1, min(int(max_depth or 1), 5))

        cypher = f"""
        MATCH p = (s:Symbol {{name: $symbol}})-[r:CALLS*1..{depth}]->(target:Symbol)
        WHERE $repo_id IS NULL OR (s.repo_id = $repo_id AND target.repo_id = $repo_id)
        RETURN DISTINCT target.name AS symbol,
               // Capitalised to match the in-memory backend and the UI's colour map. Returning
               // the raw stored type made the two backends answer the same question differently.
               coalesce(
                   CASE WHEN target.type IS NULL OR target.type = '' THEN NULL
                        ELSE toUpper(left(target.type, 1)) + substring(target.type, 1) END,
                   'Symbol'
               ) AS label,
               target.file_path AS file_path, length(p) AS depth
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, symbol=symbol, repo_id=repository_id)
                return [record.data() for record in result]
        except Exception as e:
            logger.error("Neo4j Cypher dependency query failed: %s", e)
            return []

    @staticmethod
    def _visual_payload(nodes, edges, total_nodes, total_edges, limit) -> Dict[str, Any]:
        """
        The response shape, built the same way for both backends.

        `truncated` is what the UI needs in order to stop presenting a fifth of a graph as the
        graph; the totals are what make the statement specific enough to act on.
        """
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "shown_nodes": len(nodes),
            "shown_edges": len(edges),
            "truncated": total_nodes > len(nodes),
            "node_limit": limit,
        }

    # Nodes returned to the graph view. A repository of any size exceeds what a force-directed
    # layout can render usefully, so a cap is necessary — being silent about it is not.
    MAX_VISUAL_NODES = 200

    def get_visual_graph_data(self, repository_id: str, limit: int = None) -> Dict[str, Any]:
        """
        Returns a renderable subgraph, and says how much of the whole it represents.

        Two things were wrong with the previous version, and the second is the worse one:

        1. **It was silent.** A 947-node repository rendered 200 nodes with nothing in the
           response or the UI to say so, so a developer studying the dependency graph of their
           project was looking at a fifth of it and had no way to know.
        2. **The 200 were arbitrary.** `LIMIT` with no `ORDER BY` returns whatever the store
           hands back first. The slice is now ordered by degree, so what survives truncation is
           the most connected part of the graph — the hubs a dependency view exists to show —
           rather than an accident of storage order.

        Edges are fetched *after* the nodes and restricted to that set, instead of being limited
        independently and then filtered, which silently discarded edges between nodes that were
        both present.
        """
        limit = limit or self.MAX_VISUAL_NODES

        if self.is_fallback:
            return self.fallback.get_entire_graph_data(repository_id, limit)

        # Nodes must be keyed the same way edges reference them. This previously returned the
        # internal id(n) for nodes while edges referenced a.name, so every edge was unresolvable.
        # File nodes carry 'path' and Symbol nodes carry 'name', hence the coalesce on both sides.
        # `id` carries the file so two same-named symbols are distinct nodes, while `label`
        # reports the entity kind rather than the generic :Symbol — the UI colours by label, and
        # returning labels(n)[0] made every node one colour on Neo4j but not on the fallback.
        cypher_totals = """
        MATCH (n) WHERE n.repo_id = $repo_id
        WITH count(n) AS total_nodes
        MATCH (a)-[r]->(b) WHERE a.repo_id = $repo_id AND b.repo_id = $repo_id
        RETURN total_nodes, count(r) AS total_edges
        """
        cypher_nodes = """
        MATCH (n) WHERE n.repo_id = $repo_id
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        ORDER BY degree DESC, coalesce(n.name, n.path) ASC
        LIMIT $limit
        RETURN coalesce(n.file_path, n.path, '') + '::' + coalesce(n.name, n.path) AS id,
               // Capitalised to match the fallback and the UI's colour map, which key on
               // 'Function' / 'Class' / 'Method' rather than the stored lowercase entity type.
               coalesce(
                 toUpper(substring(n.type, 0, 1)) + substring(n.type, 1),
                 labels(n)[0]
               ) AS label,
               coalesce(n.name, n.path) AS name,
               coalesce(n.file_path, n.path, '') AS file_path,
               degree
        """
        cypher_edges = """
        MATCH (a)-[r]->(b)
        WHERE a.repo_id = $repo_id AND b.repo_id = $repo_id
          AND coalesce(a.file_path, a.path, '') + '::' + coalesce(a.name, a.path) IN $ids
          AND coalesce(b.file_path, b.path, '') + '::' + coalesce(b.name, b.path) IN $ids
        RETURN coalesce(a.file_path, a.path, '') + '::' + coalesce(a.name, a.path) AS source,
               coalesce(b.file_path, b.path, '') + '::' + coalesce(b.name, b.path) AS target,
               type(r) AS type
        """

        try:
            with self.driver.session() as session:
                totals = session.run(cypher_totals, repo_id=repository_id).single()
                total_nodes = totals["total_nodes"] if totals else 0
                total_edges = totals["total_edges"] if totals else 0

                nodes = [r.data() for r in session.run(
                    cypher_nodes, repo_id=repository_id, limit=limit)]
                node_ids = [n["id"] for n in nodes]
                edges = [r.data() for r in session.run(
                    cypher_edges, repo_id=repository_id, ids=node_ids)] if node_ids else []

            return self._visual_payload(nodes, edges, total_nodes, total_edges, limit)
        except Exception as e:
            logger.error("Failed to fetch visual graph: %s", e)
            return {"nodes": [], "edges": []}

neo4j_client = Neo4jGraphClient()
