import os
import re
from typing import List, Dict, Any, Optional
from app.graph.neo4j_client import neo4j_client

class ChangeImpactAnalyzer:
    """
    Performs multi-hop reverse call graph analysis via Neo4j to determine what will be affected
    when a developer modifies a function or class. Categorizes findings into Confirmed and Inferred impacts.
    """

    @classmethod
    def analyze_change_impact(
        cls,
        target_symbol: str,
        all_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # 1. Locate target symbol chunk
        target_chunk = None
        for chunk in all_chunks:
            if target_symbol.lower() in chunk["symbol"].lower():
                target_chunk = chunk
                break

        target_file = target_chunk["file_path"] if target_chunk else "Unknown file"
        start_line = target_chunk["start_line"] if target_chunk else 1
        end_line = target_chunk["end_line"] if target_chunk else 1

        # 2. Find direct callers via the graph, scoped to this repository. Without the scope a
        # shared Neo4j instance answers with callers from other indexed repositories.
        repository_id = all_chunks[0].get("repository_id") if all_chunks else None
        callers = neo4j_client.get_callers(target_symbol, repository_id)

        confirmed_affected_files = set()
        confirmed_callers = []
        
        for caller in callers:
            caller_sym = caller.get("caller_symbol")
            fpath = caller.get("file_path")
            if fpath:
                confirmed_affected_files.add(fpath)
            confirmed_callers.append({
                "symbol": caller_sym,
                "file_path": fpath,
                "relationship": caller.get("relationship", "CALLS")
            })

        # Add target file itself
        confirmed_affected_files.add(target_file)

        # 3. Find affected test files
        affected_tests = []
        for chunk in all_chunks:
            fpath = chunk["file_path"].lower()
            code = chunk["code_snippet"].lower()
            if "test" in fpath or "spec" in fpath:
                if target_symbol.lower() in code or any(c["symbol"].lower() in code for c in confirmed_callers if c.get("symbol")):
                    affected_tests.append({
                        "file_path": chunk["file_path"],
                        "test_symbol": chunk["symbol"],
                        "lines": f"{chunk['start_line']}-{chunk['end_line']}"
                    })

        # 4. Inferred impacts, from signals that actually imply coupling.
        #
        # This previously reported every symbol sharing the target's TOP-LEVEL folder, with the
        # reason "Shares same module (backend)". In any repository where most code sits under one
        # root that flags essentially the whole codebase, which is worse than reporting nothing:
        # it buries the confirmed callers in noise and trains the reader to ignore the section.
        inferred_impacts = cls._infer_indirect_impacts(
            direct_callers=confirmed_callers,
            target_file=target_file,
            all_chunks=all_chunks,
            already_reported=confirmed_affected_files,
            repository_id=repository_id,
        )

        return {
            "primary_target": {
                "symbol": target_symbol,
                "file_path": target_file,
                "lines": f"{start_line}-{end_line}"
            },
            "confirmed_callers": confirmed_callers,
            "confirmed_files": list(confirmed_affected_files),
            "affected_tests": affected_tests,
            "inferred_impacts": inferred_impacts[:cls.MAX_INFERRED]
        }

    # Maximum indirect impacts reported. Past this the list stops being reviewable.
    MAX_INFERRED = 8
    MAX_TRANSITIVE_DEPTH = 3

    @classmethod
    def _infer_indirect_impacts(
        cls,
        direct_callers: List[Dict[str, Any]],
        target_file: str,
        all_chunks: List[Dict[str, Any]],
        already_reported: set,
        repository_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds code likely affected indirectly, from two signals that genuinely imply coupling,
        each reported with its confidence and the reason it was flagged.
        """
        impacts: List[Dict[str, Any]] = []
        seen = set()

        def record(symbol, file_path, reason, confidence):
            key = (symbol, file_path)
            if not symbol or key in seen:
                return
            seen.add(key)
            impacts.append({
                "symbol": symbol,
                "file_path": file_path,
                "reason": reason,
                "confidence": confidence,
            })

        # Signal 1: transitive callers — callers of the callers. A change that breaks a direct
        # caller propagates to whatever depends on it, so these are the strongest indirect signal.
        visited = {caller.get("symbol") for caller in direct_callers if caller.get("symbol")}
        frontier = list(visited)

        for hop in range(2, cls.MAX_TRANSITIVE_DEPTH + 1):
            next_frontier = []
            for symbol in frontier:
                for caller in neo4j_client.get_callers(symbol, repository_id):
                    caller_symbol = caller.get("caller_symbol")
                    if not caller_symbol or caller_symbol in visited:
                        continue
                    visited.add(caller_symbol)
                    next_frontier.append(caller_symbol)
                    record(
                        caller_symbol,
                        caller.get("file_path", ""),
                        f"Reaches the change in {hop} hops (via {symbol})",
                        "high" if hop == 2 else "medium",
                    )
            frontier = next_frontier
            if not frontier:
                break

        # Signal 2: modules importing the file that defines the target. Coupled through the
        # import even where no direct call edge was resolved.
        module_name = os.path.splitext(os.path.basename(target_file))[0]
        if module_name:
            for chunk in all_chunks:
                file_path = chunk.get("file_path", "")
                if file_path in already_reported or file_path == target_file:
                    continue
                for imported in chunk.get("imports") or []:
                    if module_name in re.split(r"[./\\]", imported):
                        record(
                            chunk.get("symbol", ""),
                            file_path,
                            f"Imports '{module_name}', the module being changed",
                            "medium",
                        )
                        break

        order = {"high": 0, "medium": 1, "low": 2}
        impacts.sort(key=lambda i: order.get(i["confidence"], 3))
        return impacts
