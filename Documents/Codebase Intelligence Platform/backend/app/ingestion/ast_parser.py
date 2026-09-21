import ast
import re
from typing import List, Dict, Any, Optional

class CodeEntity:
    def __init__(
        self,
        repository_id: str,
        file_path: str,
        language: str,
        entity_type: str,  # "function", "class", "method", "component", "endpoint", "table", "module"
        symbol: str,
        start_line: int,
        end_line: int,
        parent_symbol: Optional[str] = None,
        imports: Optional[List[str]] = None,
        calls: Optional[List[str]] = None,
        code_snippet: str = "",
        http_calls: Optional[List[str]] = None,
        route: Optional[str] = None
    ):
        self.repository_id = repository_id
        self.file_path = file_path
        self.language = language
        self.entity_type = entity_type
        self.symbol = symbol
        self.start_line = start_line
        self.end_line = end_line
        self.parent_symbol = parent_symbol
        self.imports = imports or []
        self.calls = calls or []
        self.code_snippet = code_snippet
        # Outbound HTTP requests as "METHOD /path", used to link frontend callers to the
        # backend endpoints that serve them (F-35).
        self.http_calls = http_calls or []
        # The route this entity serves, as "METHOD /path", when it is an endpoint. Held as an
        # attribute rather than as a separate entity so one handler is one graph node (F-29).
        self.route = route

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "file_path": self.file_path,
            "language": self.language,
            "entity_type": self.entity_type,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_symbol": self.parent_symbol,
            "imports": self.imports,
            "calls": self.calls,
            "code_snippet": self.code_snippet,
            "http_calls": self.http_calls,
            "route": self.route
        }

class MultiLanguageASTParser:
    """
    Syntax-aware parser extracting functions, classes, components, API routes, database tables, and calls.
    Preserves exact line numbers and symbol relationships.
    """

    @classmethod
    def parse_file(cls, file_meta: Dict[str, Any]) -> List[CodeEntity]:
        language = file_meta.get("language", "")
        content = file_meta.get("content", "")
        file_path = file_meta.get("relative_path", "")
        repo_id = file_meta.get("repository_id", "")

        entities: List[CodeEntity] = []

        if language == "python":
            entities.extend(cls._parse_python(content, file_path, repo_id))
        elif language in ["javascript", "typescript", "react_jsx", "react_tsx"]:
            entities.extend(cls._parse_js_ts(content, file_path, repo_id, language))
        elif language == "java":
            entities.extend(cls._parse_java(content, file_path, repo_id))
        elif language == "sql":
            entities.extend(cls._parse_sql(content, file_path, repo_id))
        else:
            # File-level entity fallback
            lines = content.splitlines()
            entities.append(CodeEntity(
                repository_id=repo_id,
                file_path=file_path,
                language=language,
                entity_type="module",
                symbol=file_path,
                start_line=1,
                end_line=max(1, len(lines)),
                code_snippet=content[:500]
            ))

        return entities

    @classmethod
    def _parse_python(cls, content: str, file_path: str, repo_id: str) -> List[CodeEntity]:
        entities = []
        lines = content.splitlines()

        try:
            tree = ast.parse(content)

            # Resolved once per file; routes are qualified with it so endpoints in different
            # routers stay distinct.
            router_prefix = cls._detect_router_prefix(content)

            # Module level imports
            module_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module_imports.append(f"{node.module or ''}.{node.names[0].name}")

            # Walk the tree rather than only tree.body.
            #
            # Iterating the module body alone skipped everything nested: closures and decorator
            # inner functions, and nested classes such as Django's `class Meta` or Pydantic's
            # `class Config`. Those are ordinary Python, and being absent from the index meant
            # they could not be found, cited, or reached by dependency analysis.
            def walk_scope(body, enclosing: str = "", enclosing_is_class: bool = False):
                for node in body:
                    if isinstance(node, ast.ClassDef):
                        qualified = f"{enclosing}.{node.name}" if enclosing else node.name
                        entities.append(CodeEntity(
                            repository_id=repo_id,
                            file_path=file_path,
                            language="python",
                            entity_type="class",
                            symbol=qualified,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            parent_symbol=enclosing or None,
                            imports=module_imports,
                            code_snippet="\n".join(lines[node.lineno - 1 : node.end_lineno])
                        ))
                        walk_scope(node.body, qualified, enclosing_is_class=True)

                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualified = f"{enclosing}.{node.name}" if enclosing else node.name

                        # A decorated handler is ONE entity, typed as an endpoint and carrying
                        # its route. Emitting a separate "POST /x" entity alongside the function
                        # split the call graph across two nodes covering identical lines:
                        # in-process callers attached to the function while ROUTES_TO attached to
                        # the route, so impact analysis missed half the dependents of each (F-29).
                        route = cls._detect_python_endpoint(node, router_prefix)

                        # "method" means "defined in a class" — a closure nested inside another
                        # function is still a function.
                        if route:
                            entity_type = "endpoint"
                        elif enclosing_is_class:
                            entity_type = "method"
                        else:
                            entity_type = "function"

                        entities.append(CodeEntity(
                            repository_id=repo_id,
                            file_path=file_path,
                            language="python",
                            entity_type=entity_type,
                            symbol=qualified,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            parent_symbol=enclosing or None,
                            imports=module_imports,
                            calls=cls._find_python_calls(node),
                            code_snippet="\n".join(lines[node.lineno - 1 : node.end_lineno]),
                            route=route
                        ))
                        walk_scope(node.body, qualified, enclosing_is_class=False)

            walk_scope(tree.body)
        except Exception:
            # Fallback regex parser for unparseable syntax
            entities.extend(cls._parse_python_regex(content, file_path, repo_id))

        return entities

    @staticmethod
    def _find_python_calls(func_node) -> List[str]:
        calls = []
        for child in ast.walk(func_node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(f"{child.func.attr}")
        return list(set(calls))

    HTTP_METHODS = ("POST", "GET", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS")

    @staticmethod
    def _detect_router_prefix(content: str) -> str:
        """
        Extracts the prefix from a module-level `APIRouter(prefix="/x")` or Flask
        `Blueprint(..., url_prefix="/x")`.

        Without it, every `@router.post("")` in the codebase produces the identical symbol, so
        unrelated endpoints in different routers collapse into one graph node. Only the local
        router is resolvable here — a prefix applied later via `include_router(...)` or
        `register_blueprint(bp, url_prefix=...)` lives in another module and is not visible at
        parse time.
        """
        match = re.search(r'APIRouter\s*\([^)]*prefix\s*=\s*[\'"]([^\'"]*)[\'"]', content)
        if match:
            return match.group(1).rstrip("/")

        # Flask's equivalent. Matched on url_prefix specifically: Blueprint's second positional
        # argument is __name__, not a path, so a looser pattern would capture the module name.
        match = re.search(r'Blueprint\s*\((?:[^)]*?)url_prefix\s*=\s*[\'"]([^\'"]*)[\'"]', content)
        return match.group(1).rstrip("/") if match else ""

    @classmethod
    def _detect_python_endpoint(cls, node, router_prefix: str = "") -> Optional[str]:
        """
        Builds a route symbol like "POST /repositories/upload" from a routing decorator.

        Handles three shapes, all of which occur in ordinary code:

        - `@router.post("/x")` / `@app.get("/x")` — FastAPI, and Flask 2.0's method shorthand,
          which happen to be identical
        - `@app.route("/x")` — classic Flask. Defaults to **GET**, as Flask does
        - `@app.route("/x", methods=["GET", "POST"])` — rendered as `"GET|POST /x"`

        The multi-method form deliberately stays **one** entity. Emitting one per method would
        split the call graph across nodes covering identical lines, which is the same mistake
        that separate route-and-function entities made (see the handler comment in `_parse_python`)
        — so the methods are carried in the route string and the graph indexes the handler under
        each of them.

        Previously an empty route produced the symbol "POST " — a trailing space and no path —
        which both read as broken in answers and collided with every other empty-route endpoint.
        """
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
                continue

            attr = decorator.func.attr
            if attr == "route":
                methods = cls._flask_route_methods(decorator)
            elif attr.upper() in cls.HTTP_METHODS:
                methods = [attr.upper()]
            else:
                continue

            if not methods:
                continue
            method = "|".join(methods)

            route = ""
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                if isinstance(decorator.args[0].value, str):
                    route = decorator.args[0].value

            full_path = f"{router_prefix}{route}"
            if not full_path.startswith("/"):
                full_path = f"/{full_path}"

            # Normalises "" -> "/" and strips a trailing slash from "/repositories/".
            full_path = full_path.rstrip("/") or "/"

            return f"{method} {full_path}"
        return None

    @classmethod
    def _flask_route_methods(cls, decorator) -> List[str]:
        """
        The HTTP methods a Flask `@route` decorator responds to.

        Flask defaults to GET when `methods=` is absent, which is why a bare `@app.route("/x")`
        must not be skipped — it is a real GET endpoint, and skipping it was the whole of this
        defect. Sorted so an index is reproducible regardless of the order they were written in.
        """
        for keyword in decorator.keywords:
            if keyword.arg != "methods":
                continue
            if not isinstance(keyword.value, (ast.List, ast.Tuple, ast.Set)):
                # A variable or comprehension: the methods are not knowable statically, and
                # guessing GET would be wrong. Treat it as a route of unknown method.
                return []
            methods = [
                element.value.upper()
                for element in keyword.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
                and element.value.upper() in cls.HTTP_METHODS
            ]
            return sorted(set(methods))

        return ["GET"]

    @classmethod
    def _parse_python_regex(cls, content: str, file_path: str, repo_id: str) -> List[CodeEntity]:
        entities = []
        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            func_match = re.match(r'^\s*(async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(', line)
            if func_match:
                symbol = func_match.group(2)
                entities.append(CodeEntity(
                    repository_id=repo_id,
                    file_path=file_path,
                    language="python",
                    entity_type="function",
                    symbol=symbol,
                    start_line=idx,
                    end_line=min(idx + 25, len(lines)),
                    code_snippet="\n".join(lines[idx - 1 : idx + 20])
                ))
        return entities

    @classmethod
    def _parse_js_ts(cls, content: str, file_path: str, repo_id: str, language: str) -> List[CodeEntity]:
        # Prefer real syntax trees; the regex path below is the fallback when tree-sitter is
        # unavailable and cannot determine where a definition actually ends.
        from app.ingestion.treesitter_parser import TreeSitterParser
        parsed = TreeSitterParser.parse(content, file_path, repo_id, language)
        if parsed is not None:
            return parsed

        return cls._parse_js_ts_regex(content, file_path, repo_id, language)

    @classmethod
    def _parse_js_ts_regex(cls, content: str, file_path: str, repo_id: str, language: str) -> List[CodeEntity]:
        """Approximate fallback: line extents are guessed, not derived. See F-25."""
        entities = []
        lines = content.splitlines()

        # Imports
        imports = re.findall(r'import\s+.*?from\s+[\'"](.*?)[\'"]', content)

        # Functions / Arrow functions
        func_pattern = re.compile(r'(export\s+)?(async\s+)?function\s+([a-zA-Z0-9_]+)\s*\(')
        arrow_pattern = re.compile(r'(export\s+)?const\s+([a-zA-Z0-9_]+)\s*=\s*(async\s*)?\(')

        # API fetch calls
        api_calls = re.findall(r'fetch\([\'"`](.*?)[\'"`]', content)

        for idx, line in enumerate(lines, 1):
            m_func = func_pattern.search(line)
            m_arrow = arrow_pattern.search(line)

            symbol = None
            is_component = False

            if m_func:
                symbol = m_func.group(3)
            elif m_arrow:
                symbol = m_arrow.group(2)

            if symbol:
                if symbol[0].isupper() or "Component" in symbol or "JSX" in language.upper():
                    is_component = True

                end_l = min(idx + 30, len(lines))
                snippet = "\n".join(lines[idx - 1 : end_l])

                entity_type = "component" if is_component else "function"

                entities.append(CodeEntity(
                    repository_id=repo_id,
                    file_path=file_path,
                    language=language,
                    entity_type=entity_type,
                    symbol=symbol,
                    start_line=idx,
                    end_line=end_l,
                    imports=imports,
                    calls=api_calls,
                    code_snippet=snippet
                ))

        if not entities:
            entities.append(CodeEntity(
                repository_id=repo_id,
                file_path=file_path,
                language=language,
                entity_type="module",
                symbol=file_path,
                start_line=1,
                end_line=max(1, len(lines)),
                code_snippet=content[:500]
            ))

        return entities

    @classmethod
    def _parse_java(cls, content: str, file_path: str, repo_id: str) -> List[CodeEntity]:
        from app.ingestion.treesitter_parser import TreeSitterParser
        parsed = TreeSitterParser.parse(content, file_path, repo_id, "java")
        if parsed is not None:
            return parsed

        return cls._parse_java_regex(content, file_path, repo_id)

    @classmethod
    def _parse_java_regex(cls, content: str, file_path: str, repo_id: str) -> List[CodeEntity]:
        """Approximate fallback: line extents are guessed, and no calls are extracted. See F-25."""
        entities = []
        lines = content.splitlines()

        class_match = re.search(r'public\s+class\s+([a-zA-Z0-9_]+)', content)
        parent_class = class_match.group(1) if class_match else file_path

        method_pattern = re.compile(r'public\s+([a-zA-Z0-9_<>\s]+)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{')
        for idx, line in enumerate(lines, 1):
            m = method_pattern.search(line)
            if m:
                method_name = m.group(2)
                end_l = min(idx + 35, len(lines))
                entities.append(CodeEntity(
                    repository_id=repo_id,
                    file_path=file_path,
                    language="java",
                    entity_type="method",
                    symbol=f"{parent_class}.{method_name}",
                    start_line=idx,
                    end_line=end_l,
                    parent_symbol=parent_class,
                    code_snippet="\n".join(lines[idx - 1 : end_l])
                ))

        return entities

    @classmethod
    def _parse_sql(cls, content: str, file_path: str, repo_id: str) -> List[CodeEntity]:
        entities = []
        tables = re.findall(r'CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+)', content, re.IGNORECASE)
        lines = content.splitlines()

        for t in tables:
            table_name = t[1]
            entities.append(CodeEntity(
                repository_id=repo_id,
                file_path=file_path,
                language="sql",
                entity_type="table",
                symbol=f"table:{table_name}",
                start_line=1,
                end_line=len(lines),
                code_snippet=content[:800]
            ))
        return entities
