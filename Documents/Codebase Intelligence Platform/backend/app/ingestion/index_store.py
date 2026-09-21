import os
import json
import logging
from typing import List, Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger("ingestion.index_store")


class RepositoryIndexStore:
    """
    Persists a repository's parsed chunks to disk so the index survives the process that built it.

    Previously the only copy lived in a module-level dict, which meant the whole corpus was
    re-parsed from source on every startup (blocking the event loop while it ran) and a second
    uvicorn worker saw an empty index for anything the first had ingested.

    One JSON file per repository. `formatted_content` is derived from the other fields, so it is
    dropped on save and rebuilt on load rather than stored twice.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(settings.DATA_DIR, "indexes")

    def _path(self, repository_id: str) -> str:
        safe_id = repository_id.replace("/", "_").replace("..", "_")
        return os.path.join(self.base_dir, f"{safe_id}.json")

    def exists(self, repository_id: str) -> bool:
        return os.path.isfile(self._path(repository_id))

    def save(self, repository_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Writes the index atomically, so an interrupted save cannot leave a partial file."""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            slim = [{k: v for k, v in chunk.items() if k != "formatted_content"} for chunk in chunks]

            target = self._path(repository_id)
            tmp_path = f"{target}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(slim, f, default=str)
            os.replace(tmp_path, target)
            return True
        except Exception as e:
            logger.error("Could not persist index for '%s': %s", repository_id, e)
            return False

    def load(self, repository_id: str) -> Optional[List[Dict[str, Any]]]:
        path = self._path(repository_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not isinstance(chunks, list):
                raise ValueError("index file is not a list of chunks")

            from app.ingestion.chunker import HierarchicalCodeChunker
            for chunk in chunks:
                chunk["formatted_content"] = HierarchicalCodeChunker.format_chunk_content(chunk)
            return chunks
        except Exception as e:
            # A corrupt index must not be fatal: the caller falls back to re-indexing from source.
            logger.error("Could not load index for '%s' (%s); it will be rebuilt.", repository_id, e)
            return None

    def _manifest_path(self, repository_id: str) -> str:
        safe_id = repository_id.replace("/", "_").replace("..", "_")
        return os.path.join(self.base_dir, f"{safe_id}.manifest.json")

    def save_manifest(self, repository_id: str, manifest: Dict[str, str]) -> bool:
        """Stores `{relative_path: file_hash}` so the next index can skip unchanged files."""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            target = self._manifest_path(repository_id)
            tmp_path = f"{target}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            os.replace(tmp_path, target)
            return True
        except Exception as e:
            logger.error("Could not persist manifest for '%s': %s", repository_id, e)
            return False

    def load_manifest(self, repository_id: str) -> Dict[str, str]:
        path = self._manifest_path(repository_id)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return manifest if isinstance(manifest, dict) else {}
        except Exception as e:
            # Without a manifest every file is treated as changed, which is correct but slower.
            logger.error("Could not load manifest for '%s' (%s); re-indexing in full.", repository_id, e)
            return {}

    def delete(self, repository_id: str) -> None:
        """
        Removes the chunk index **and** its manifest.

        Deleting only the index left the manifest behind, and the manifest is what the incremental
        path trusts to decide a file is unchanged (F-23). Re-creating a repository under the same
        id then matched every hash against the orphaned manifest, concluded nothing had changed,
        and reused chunks from an index that no longer existed — producing an empty repository
        that reported itself as indexed.
        """
        for path in (self._path(repository_id), self._manifest_path(repository_id)):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error("Could not delete %s for '%s': %s", path, repository_id, e)


index_store = RepositoryIndexStore()
