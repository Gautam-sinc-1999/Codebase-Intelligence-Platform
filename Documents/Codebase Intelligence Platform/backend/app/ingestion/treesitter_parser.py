import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger("ingestion.treesitter")


class TreeSitterParser:
    """
    Syntax-tree parser for JavaScript, TypeScript, JSX/TSX and Java.

    These languages were previously parsed line-by-line with regular expressions, which could
    not determine where a definition ended — extents were invented as `start + 30` (JS) and
    `start + 35` (Java). Every line range shown for non-Python code was therefore a guess, and
    the product's "exact line numbers" claim held only for Python. Regexes also missed class
    bodies, nested functions, object-literal methods and default exports entirely, and extracted
    no call edges at all for Java while extracting only `fetch()` URL strings for JS — leaving
    the call graph effectively Python-only.

    Tree-sitter gives real node extents and real call expressions. If the dependency is absent
    the caller falls back to the previous regex parser, so this stays optional.
    """

    # Our internal language ids mapped onto grammar names. JSX is handled by the javascript
    # grammar; TSX needs its own because the generics syntax is ambiguous with JSX elements.
    GRAMMARS = {
        "javascript": "javascript",
        "react_jsx": "javascript",
        "typescript": "typescript",
        "react_tsx": "tsx",
        "java": "java",
    }

    _parsers: Dict[str, Any] = {}
    _available: Optional[bool] = None

    @classmethod
    def available(cls) -> bool:
        if cls._available is None:
            try:
                import tree_sitter_language_pack  # noqa: F401
                cls._available = True
            except ImportError:
                logger.warning(
                    "tree-sitter not installed; JS/TS/Java will use the regex parser, whose "
                    "line ranges are approximate."
                )
                cls._available = False
        return cls._available

    @classmethod
    def _parser_for(cls, language: str):
        grammar = cls.GRAMMARS.get(language)
        if not grammar:
            return None
        if grammar not in cls._parsers:
            try:
                from tree_sitter_language_pack import get_parser
                cls._parsers[grammar] = get_parser(grammar)
            except Exception as e:
                logger.error("Could not load '%s' grammar: %s", grammar, e)
                cls._parsers[grammar] = None
        return cls._parsers[grammar]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _text(node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _lines(node) -> tuple:
        """Real 1-based inclusive line range, straight from the node's extent."""
        return node.start_point[0] + 1, node.end_point[0] + 1

    @classmethod
    def _named_child_text(cls, node, field: str, source: bytes) -> str:
        child = node.child_by_field_name(field)
        return cls._text(child, source) if child else ""

    @classmethod
    def _callee_name(cls, node, source: bytes) -> str:
        """
        Resolves the callee of a call expression to a bare name.

        Mirrors the Python parser, which records the final attribute of an attribute call, so
        both languages feed the graph the same shape of name and resolve through the same
        definition index at graph-build time.
        """
        if node is None:
            return ""
        if node.type in ("identifier", "property_identifier", "type_identifier", "shorthand_property_identifier"):
            return cls._text(node, source)
        if node.type in ("member_expression", "field_access"):
            prop = node.child_by_field_name("property") or node.child_by_field_name("field")
            return cls._text(prop, source) if prop else ""
        if node.type == "scoped_identifier":
            name = node.child_by_field_name("name")
            return cls._text(name, source) if name else ""
        if node.type in ("generic_type", "scoped_type_identifier"):
            for child in node.children:
                if child.type in ("type_identifier", "identifier"):
                    return cls._text(child, source)
        return ""

    CALL_NODES = {"call_expression", "method_invocation", "new_expression", "object_creation_expression"}

    # Rendering <Card/> is how one React component depends on another, but JSX elements are not
    # call expressions, so component composition — the actual structure of a React application —
    # was entirely absent from the call graph. "What depends on Card?" answered nothing.
    JSX_ELEMENT_NODES = {"jsx_opening_element", "jsx_self_closing_element"}

    @classmethod
    def _jsx_component_name(cls, node, source: bytes) -> str:
        """
        Returns the component name for a JSX element, or "" for a plain HTML tag.

        React's convention is the discriminator: <Card/> is a component reference, <div/> is not.
        Member expressions such as <Modal.Header/> resolve to the final property, matching how
        every other callee is recorded.
        """
        if node is None:
            return ""
        if node.type == "member_expression":
            prop = node.child_by_field_name("property")
            text = cls._text(prop, source) if prop else ""
        elif node.type in ("identifier", "jsx_identifier"):
            text = cls._text(node, source)
        else:
            return ""
        return text if text[:1].isupper() else ""

    @classmethod
    def _collect_calls(cls, node, source: bytes) -> List[str]:
        """Walks a definition's subtree collecting every invocation target."""
        calls: List[str] = []
        seen: Set[str] = set()
        stack = [node]

        while stack:
            current = stack.pop()

            if current.type in cls.JSX_ELEMENT_NODES:
                # Only capitalised names are components; lowercase tags are HTML elements.
                element = current.child_by_field_name("name")
                name = cls._jsx_component_name(element, source)
                if name and name not in seen:
                    seen.add(name)
                    calls.append(name)

            elif current.type in cls.CALL_NODES:
                target = None
                if current.type == "call_expression":
                    target = current.child_by_field_name("function")
                elif current.type == "method_invocation":
                    target = current.child_by_field_name("name")
                elif current.type == "new_expression":
                    target = current.child_by_field_name("constructor")
                elif current.type == "object_creation_expression":
                    target = current.child_by_field_name("type")

                name = cls._callee_name(target, source)
                if name and name not in seen:
                    seen.add(name)
                    calls.append(name)

            stack.extend(current.children)

        return calls

    HTTP_VERBS = {"get", "post", "put", "delete", "patch", "head", "options"}

    @classmethod
    def _literal_path(cls, node, source: bytes) -> str:
        """
        Reads a URL argument. Template literals keep their static structure with interpolations
        replaced by `*`, so `/orders/${id}` becomes `/orders/*` and can still match a route
        declared as `/orders/{order_id}`.
        """
        if node is None:
            return ""

        if node.type == "string":
            return cls._text(node, source).strip("\"'`")

        if node.type == "template_string":
            out = []
            for child in node.children:
                if child.type in ("template_substitution", "template_literal_substitution"):
                    out.append("*")
                else:
                    out.append(cls._text(child, source))
            return "".join(out).strip("`")

        return ""

    @classmethod
    def _collect_http_calls(cls, node, source: bytes) -> List[str]:
        """
        Extracts outbound HTTP requests as "METHOD /path" — the same shape backend endpoint
        symbols use — so the graph can join a component to the route that serves it.

        Recognises `fetch(url, { method })`, which defaults to GET, and `axios.post(url)` style
        verb methods on any client object.
        """
        found: List[str] = []
        seen: Set[str] = set()
        stack = [node]

        while stack:
            current = stack.pop()
            stack.extend(current.children)

            if current.type != "call_expression":
                continue

            function = current.child_by_field_name("function")
            arguments = current.child_by_field_name("arguments")
            if function is None or arguments is None:
                continue

            args = [a for a in arguments.children if a.is_named]
            if not args:
                continue

            method = ""
            url = ""

            if function.type == "identifier" and cls._text(function, source) == "fetch":
                url = cls._literal_path(args[0], source)
                method = "GET"
                # A second argument may carry `method: 'POST'`.
                if len(args) > 1 and args[1].type == "object":
                    for pair in args[1].children:
                        if pair.type != "pair":
                            continue
                        key = pair.child_by_field_name("key")
                        value = pair.child_by_field_name("value")
                        if key is not None and value is not None:
                            if cls._text(key, source).strip("\"'") == "method":
                                literal = cls._text(value, source).strip("\"'`")
                                if literal:
                                    method = literal.upper()

            elif function.type == "member_expression":
                verb = cls._callee_name(function, source).lower()
                if verb in cls.HTTP_VERBS:
                    url = cls._literal_path(args[0], source)
                    method = verb.upper()

            if not url or not url.startswith(("/", "http")):
                continue

            entry = f"{method} {url}"
            if entry not in seen:
                seen.add(entry)
                found.append(entry)

        return found

    @classmethod
    def _collect_imports(cls, root, source: bytes) -> List[str]:
        imports: List[str] = []
        seen: Set[str] = set()
        stack = [root]

        while stack:
            node = stack.pop()

            if node.type == "import_statement":  # JS/TS
                src = node.child_by_field_name("source")
                if src:
                    value = cls._text(src, source).strip("\"'`")
                    if value and value not in seen:
                        seen.add(value)
                        imports.append(value)

            elif node.type == "import_declaration":  # Java
                value = cls._text(node, source).replace("import", "").replace(";", "").strip()
                if value and value not in seen:
                    seen.add(value)
                    imports.append(value)

            stack.extend(node.children)

        return imports

    # ------------------------------------------------------------------ definitions

    # Node types that declare something worth indexing, by grammar family.
    JS_FUNCTION_NODES = {
        "function_declaration", "generator_function_declaration", "function_expression",
        "arrow_function", "method_definition", "method_signature", "abstract_method_signature",
    }
    # `abstract_class_declaration` is TypeScript-only and was absent, so an abstract class was
    # not recognised as a container at all — its methods were emitted unqualified (`helper`
    # rather than `BaseRepo.helper`), which both loses the owner and collides with every other
    # `helper` in the repository.
    JS_CLASS_NODES = {
        "class_declaration", "class", "abstract_class_declaration",
        "interface_declaration", "internal_module", "module",
    }
    JAVA_TYPE_NODES = {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}

    # TypeScript declarations that name something without being a container.
    TS_ALIAS_NODES = {"type_alias_declaration", "enum_declaration"}
    JAVA_METHOD_NODES = {"method_declaration", "constructor_declaration"}

    @classmethod
    def _binding_name(cls, node, source: bytes) -> str:
        """
        Returns the variable a function expression is bound to, looking through one wrapper call.

        Covers `const Plain = function Named() {...}` and `const Memoed = memo(function Inner(){})`.
        In both, the *binding* is what other modules import and render — `<Memoed/>` elsewhere
        must resolve here — while the inner name is only visible within the function. Taking the
        inner name left the exported component with no definition node and the inner one with
        no callers.
        """
        parent = node.parent
        if parent is None:
            return ""

        # Directly assigned: const X = function Y() {} / const X = () => {}
        if parent.type == "variable_declarator":
            return cls._named_child_text(parent, "name", source)

        # Wrapped in one call: const X = memo(...) / forwardRef(...) / observer(...)
        if parent.type == "arguments":
            call = parent.parent
            if call is not None and call.type == "call_expression":
                declarator = call.parent
                if declarator is not None and declarator.type == "variable_declarator":
                    return cls._named_child_text(declarator, "name", source)

        # Object property or assignment: { handler: function () {} } / obj.x = () => {}
        if parent.type in ("pair", "assignment_expression"):
            key = parent.child_by_field_name("key") or parent.child_by_field_name("left")
            return cls._callee_name(key, source) if key else ""

        return ""

    @classmethod
    def _definition_name(cls, node, source: bytes) -> str:
        """Finds a declaration's name, including arrow functions bound to a variable."""
        # For function *expressions* the binding wins over the inner name; a function
        # *declaration* keeps its own name, which is also its binding.
        if node.type in ("arrow_function", "function_expression"):
            bound = cls._binding_name(node, source)
            if bound:
                return bound

        name = cls._named_child_text(node, "name", source)
        if name:
            return name

        # `const Checkout = () => {...}` / `const load = async function () {...}`
        return ""

    @classmethod
    def parse(cls, content: str, file_path: str, repo_id: str, language: str) -> Optional[List]:
        """
        Returns CodeEntity objects with true line ranges, or None if tree-sitter cannot handle
        this language so the caller can fall back.
        """
        if not cls.available():
            return None

        parser = cls._parser_for(language)
        if parser is None:
            return None

        # Import here to avoid a circular import at module load.
        from app.ingestion.ast_parser import CodeEntity

        try:
            source = content.encode("utf-8")
            tree = parser.parse(source)
        except Exception as e:
            logger.error("tree-sitter failed to parse %s: %s", file_path, e)
            return None

        root = tree.root_node
        imports = cls._collect_imports(root, source)
        is_jsx = language in ("react_jsx", "react_tsx")
        entities: List[CodeEntity] = []

        def walk(node, enclosing_type: str = ""):
            node_type = node.type

            is_js_class = node_type in cls.JS_CLASS_NODES
            is_java_type = node_type in cls.JAVA_TYPE_NODES
            is_js_func = node_type in cls.JS_FUNCTION_NODES
            is_java_method = node_type in cls.JAVA_METHOD_NODES

            if is_js_class or is_java_type:
                name = cls._definition_name(node, source)
                if name:
                    # Qualify nested containers: an inner class is `Outer.Inner`, not `Inner`.
                    # Without this a Java inner class or a TS namespace member lost its owner.
                    if enclosing_type:
                        name = f"{enclosing_type}.{name}"
                    start, end = cls._lines(node)
                    # Calls are deliberately not collected at class level: every one of them
                    # belongs to a method that is indexed separately, so gathering them here
                    # too would duplicate each edge in the call graph. This matches how the
                    # Python parser treats classes.
                    entities.append(CodeEntity(
                        repository_id=repo_id, file_path=file_path, language=language,
                        entity_type="class", symbol=name, start_line=start, end_line=end,
                        imports=imports, code_snippet=cls._text(node, source),
                    ))
                    for child in node.children:
                        walk(child, name)
                    return

            elif is_js_func or is_java_method:
                name = cls._definition_name(node, source)
                if name:
                    start, end = cls._lines(node)
                    qualified = f"{enclosing_type}.{name}" if enclosing_type else name

                    if enclosing_type:
                        entity_type = "method"
                    elif is_jsx and name[:1].isupper():
                        # React components are conventionally capitalised; the regex parser used
                        # the same signal, but now over a correctly delimited definition.
                        entity_type = "component"
                    else:
                        entity_type = "function"

                    entities.append(CodeEntity(
                        repository_id=repo_id, file_path=file_path, language=language,
                        entity_type=entity_type, symbol=qualified,
                        start_line=start, end_line=end,
                        parent_symbol=enclosing_type or None,
                        imports=imports, calls=cls._collect_calls(node, source),
                        http_calls=cls._collect_http_calls(node, source),
                        code_snippet=cls._text(node, source),
                    ))
                    # Do not descend: nested helpers are already covered by the parent's extent
                    # and its collected calls.
                    return

            for child in node.children:
                walk(child, enclosing_type)

        walk(root)

        if not entities:
            line_count = len(content.splitlines())
            entities.append(CodeEntity(
                repository_id=repo_id, file_path=file_path, language=language,
                entity_type="module", symbol=file_path,
                start_line=1, end_line=max(1, line_count),
                imports=imports, code_snippet=content[:500],
            ))

        return entities
