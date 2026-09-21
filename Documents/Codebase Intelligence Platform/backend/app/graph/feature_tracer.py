from typing import List, Dict, Any

class FeatureTracer:
    """
    Traces execution flow across Frontend components, API endpoints, backend controllers, services, and database tables.
    """

    @classmethod
    def trace_feature_flow(cls, feature_keyword: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        flow_steps = []
        frontend_components = []
        api_endpoints = []
        backend_controllers = []
        backend_services = []
        database_tables = []
        relevant_files = []

        feature_term = feature_keyword.lower()

        for chunk in chunks:
            symbol = chunk.get("symbol", "")
            file_path = chunk.get("file_path", "")
            entity_type = chunk.get("entity_type", "")
            code = chunk.get("code_snippet", "").lower()
            start_line = chunk.get("start_line", 1)
            end_line = chunk.get("end_line", 1)

            if feature_term in symbol.lower() or feature_term in file_path.lower() or feature_term in code:
                ref_item = {
                    "file_path": file_path,
                    "symbol": symbol,
                    "entity_type": entity_type,
                    "start_line": start_line,
                    "end_line": end_line,
                    "snippet": chunk.get("code_snippet", "")
                }
                relevant_files.append(ref_item)

                if entity_type == "component":
                    frontend_components.append(ref_item)
                    flow_steps.append({
                        "step": len(flow_steps) + 1,
                        "layer": "Frontend Component",
                        "title": f"Component: {symbol}",
                        "file_path": file_path,
                        "lines": f"{start_line}-{end_line}",
                        "description": f"User triggers action in {symbol} ({file_path}:{start_line})"
                    })
                elif entity_type == "endpoint":
                    api_endpoints.append(ref_item)
                    # The route lives on the chunk now that a handler is a single entity (F-29),
                    # so show it alongside the handler name rather than instead of it.
                    route = chunk.get("route") or ""
                    label = f"{route} ({symbol})" if route else symbol
                    flow_steps.append({
                        "step": len(flow_steps) + 1,
                        "layer": "HTTP API Gateway",
                        "title": f"Endpoint: {label}",
                        "file_path": file_path,
                        "lines": f"{start_line}-{end_line}",
                        "description": f"HTTP request to {label} handled in {file_path}"
                    })
                elif entity_type in ["function", "method"] and ("controller" in file_path.lower() or "router" in file_path.lower()):
                    backend_controllers.append(ref_item)
                    flow_steps.append({
                        "step": len(flow_steps) + 1,
                        "layer": "Backend Controller",
                        "title": f"Controller: {symbol}",
                        "file_path": file_path,
                        "lines": f"{start_line}-{end_line}",
                        "description": f"Controller action handles request parameters at {file_path}:{start_line}"
                    })
                elif entity_type in ["function", "method", "class"] and "service" in file_path.lower():
                    backend_services.append(ref_item)
                    flow_steps.append({
                        "step": len(flow_steps) + 1,
                        "layer": "Business Service",
                        "title": f"Service: {symbol}",
                        "file_path": file_path,
                        "lines": f"{start_line}-{end_line}",
                        "description": f"Core business logic executed by {symbol} ({file_path}:{start_line})"
                    })
                elif entity_type == "table" or "model" in file_path.lower() or "schema" in file_path.lower():
                    database_tables.append(ref_item)
                    flow_steps.append({
                        "step": len(flow_steps) + 1,
                        "layer": "Database Persistence",
                        "title": f"Database / Model: {symbol}",
                        "file_path": file_path,
                        "lines": f"{start_line}-{end_line}",
                        "description": f"State saved to database table/model in {file_path}"
                    })

        # Ensure ordered fallback flow if step count is sparse
        if not flow_steps and relevant_files:
            for idx, item in enumerate(relevant_files[:5], 1):
                flow_steps.append({
                    "step": idx,
                    "layer": item["entity_type"].capitalize(),
                    "title": item["symbol"],
                    "file_path": item["file_path"],
                    "lines": f"{item['start_line']}-{item['end_line']}",
                    "description": f"Execution passes through {item['symbol']} in {item['file_path']}"
                })

        return {
            "feature": feature_keyword,
            "flow_steps": flow_steps,
            "frontend_components": frontend_components,
            "api_endpoints": api_endpoints,
            "backend_controllers": backend_controllers,
            "backend_services": backend_services,
            "database_tables": database_tables,
            "relevant_files": relevant_files
        }
