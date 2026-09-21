import hashlib
from typing import List, Dict, Any
from app.ingestion.ast_parser import CodeEntity

class HierarchicalCodeChunker:
    """
    Creates symbol-aware, semantic code chunks from AST entities and file metadata.
    Avoids arbitrary token truncation and preserves line numbers.
    """

    @staticmethod
    def format_chunk_content(chunk: Dict[str, Any]) -> str:
        """
        Builds the text a chunk is keyword-indexed on. Derived entirely from the other fields,
        so the index store rebuilds it on load instead of persisting a second copy of the code.
        """
        return (
            f"Symbol: {chunk.get('symbol', '')}\n"
            f"File: {chunk.get('file_path', '')}:{chunk.get('start_line', 1)}-{chunk.get('end_line', 1)}\n"
            f"Language: {chunk.get('language', '')}\n\n"
            f"{chunk.get('code_snippet', '')}"
        )

    @classmethod
    def create_chunks(cls, file_meta: Dict[str, Any], entities: List[CodeEntity]) -> List[Dict[str, Any]]:
        chunks = []
        repo_id = file_meta.get("repository_id", "")
        file_path = file_meta.get("relative_path", "")
        language = file_meta.get("language", "")

        for entity in entities:
            chunk_raw = f"Symbol: {entity.symbol}\nFile: {file_path}:{entity.start_line}-{entity.end_line}\nLanguage: {language}\n\n{entity.code_snippet}"
            chunk_id = hashlib.md5(f"{repo_id}:{file_path}:{entity.symbol}:{entity.start_line}".encode("utf-8")).hexdigest()

            summary = f"{entity.entity_type.capitalize()} '{entity.symbol}' in {file_path} (lines {entity.start_line}-{entity.end_line})."

            chunk_dict = {
                "chunk_id": chunk_id,
                "repository_id": repo_id,
                "file_path": file_path,
                "language": language,
                "entity_type": entity.entity_type,
                "symbol": entity.symbol,
                "start_line": entity.start_line,
                "end_line": entity.end_line,
                "parent_symbol": entity.parent_symbol,
                "imports": entity.imports,
                "calls": entity.calls,
                "http_calls": entity.http_calls,
                "route": entity.route,
                "code_snippet": entity.code_snippet,
                "formatted_content": chunk_raw,
                "summary": summary
            }

            chunks.append(chunk_dict)

        return chunks
