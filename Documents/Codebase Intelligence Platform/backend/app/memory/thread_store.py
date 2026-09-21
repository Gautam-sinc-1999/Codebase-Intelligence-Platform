import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.core.config import settings

logger = logging.getLogger("memory.thread_store")


class ThreadStore:
    """
    One folder per conversation thread.

        data/threads/<conversation_id>/
            thread.json      metadata: repository_id, title, summary, working_context, counters
            messages.jsonl   one JSON object per line, appended

    **How Neo4j and ChromaDB are reached from here:** they are not stored in the folder and are
    not rebuilt from it. `thread.json` holds `repository_id`, and that single field routes to the
    repository's index, vector collection and graph:

        thread.json.repository_id ──┬──► data/indexes/<repo>.json   (chunks)
                                    ├──► chroma collection repo_<repo>
                                    └──► neo4j nodes where repo_id = <repo>

    Reconstructing the index from the thread is not possible: messages record only the citations
    that appeared in past answers — roughly 2 % of a repository's chunks in a measured session —
    and none of the CALLS edges between everything else. The index is derived data, rebuildable
    from source in seconds; the thread is not rebuildable from anything, which is why they have
    separate lifecycles.

    Messages are JSON Lines rather than a single JSON array so appending a turn is one `write`
    with no read-modify-write of the whole history, and a thread has no practical size ceiling.
    """

    THREAD_FILE = "thread.json"
    MESSAGES_FILE = "messages.jsonl"

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(settings.DATA_DIR, "threads")

    # ------------------------------------------------------------------ paths

    def _safe_id(self, conversation_id: str) -> str:
        """Conversation ids are generated, but a folder name must never escape base_dir."""
        cleaned = os.path.basename(str(conversation_id).replace("\\", "/"))
        if cleaned in ("", ".", ".."):
            raise ValueError(f"unsafe conversation id: {conversation_id!r}")
        return cleaned

    def thread_dir(self, conversation_id: str) -> str:
        return os.path.join(self.base_dir, self._safe_id(conversation_id))

    def _meta_path(self, conversation_id: str) -> str:
        return os.path.join(self.thread_dir(conversation_id), self.THREAD_FILE)

    def _messages_path(self, conversation_id: str) -> str:
        return os.path.join(self.thread_dir(conversation_id), self.MESSAGES_FILE)

    def exists(self, conversation_id: str) -> bool:
        return os.path.isfile(self._meta_path(conversation_id))

    # ------------------------------------------------------------------ metadata

    def _write_meta(self, conversation_id: str, meta: Dict[str, Any]) -> None:
        """Written atomically so an interrupted save cannot corrupt the thread."""
        os.makedirs(self.thread_dir(conversation_id), exist_ok=True)
        target = self._meta_path(conversation_id)
        tmp_path = f"{target}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, default=str)
        os.replace(tmp_path, target)

    def read_meta(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        path = self._meta_path(conversation_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as e:
            logger.error("Could not read thread metadata for '%s': %s", conversation_id, e)
            return None

    def create(self, conversation_id: str, repository_id: str, title: str = "New Analysis") -> Dict[str, Any]:
        existing = self.read_meta(conversation_id)
        if existing:
            return existing

        now = datetime.utcnow().isoformat()
        meta = {
            "conversation_id": conversation_id,
            # The join key to the index, vector store and graph. Everything else about this
            # thread is local; this is the one pointer outward.
            "repository_id": repository_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "summary": "",
            "working_context": {"current_feature": "", "current_symbol": "", "relevant_files": []},
            "message_count": 0,
            # The repository commit this thread's answers were written against. Empty until the
            # first turn, and only ever set for a repository that has an upstream.
            #
            # A thread is a conversation about a *specific version* of the code. Once the index
            # moves on, the citations stored in earlier turns still resolve — they just resolve
            # against different code — so the version has to be recorded to be able to say so.
            "indexed_commit_sha": "",
            # A remote SHA the user has explicitly chosen to stay behind, so the drift banner is
            # shown once per new commit rather than on every message.
            "drift_ack_sha": "",
        }
        self._write_meta(conversation_id, meta)
        open(self._messages_path(conversation_id), "a", encoding="utf-8").close()
        return meta

    def update_meta(self, conversation_id: str, **fields) -> Optional[Dict[str, Any]]:
        meta = self.read_meta(conversation_id)
        if meta is None:
            return None
        meta.update(fields)
        meta["updated_at"] = datetime.utcnow().isoformat()
        self._write_meta(conversation_id, meta)
        return meta

    # ------------------------------------------------------------------ messages

    def append_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> int:
        """Appends turns to the JSONL log and returns the new total."""
        if not messages:
            return self.read_meta(conversation_id).get("message_count", 0) if self.exists(conversation_id) else 0

        os.makedirs(self.thread_dir(conversation_id), exist_ok=True)
        with open(self._messages_path(conversation_id), "a", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(message, default=str) + "\n")

        meta = self.read_meta(conversation_id) or {}
        total = int(meta.get("message_count", 0)) + len(messages)
        self.update_meta(conversation_id, message_count=total)
        return total

    def read_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Reads the thread's messages, optionally only the most recent `limit`."""
        path = self._messages_path(conversation_id)
        if not os.path.isfile(path):
            return []

        messages: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        # One bad line must not make the whole thread unreadable, which is a
                        # concrete advantage of JSONL over a single JSON array.
                        logger.warning("Skipping unreadable message line in '%s'", conversation_id)
        except Exception as e:
            logger.error("Could not read messages for '%s': %s", conversation_id, e)
            return []

        return messages[-limit:] if limit else messages

    def load(self, conversation_id: str, message_limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Returns the full thread — metadata plus messages — in the shape the API returns."""
        meta = self.read_meta(conversation_id)
        if meta is None:
            return None
        thread = dict(meta)
        thread["messages"] = self.read_messages(conversation_id, limit=message_limit)
        return thread

    # ------------------------------------------------------------------ listing

    def list_threads(self, repository_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lists threads newest first, reading only metadata.

        This is the cost of folder-per-thread: the sidebar reads one small file per thread
        rather than issuing one indexed query. Fine at hundreds of threads; if it becomes the
        bottleneck, the fix is an index file at base_dir rather than a different store.
        """
        if not os.path.isdir(self.base_dir):
            return []

        threads = []
        for entry in os.listdir(self.base_dir):
            meta = self.read_meta(entry)
            if meta is None:
                continue
            if repository_id and meta.get("repository_id") != repository_id:
                continue
            threads.append(meta)

        threads.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
        return threads[:limit]

    def delete(self, conversation_id: str) -> bool:
        directory = self.thread_dir(conversation_id)
        if not os.path.isdir(directory):
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True


thread_store = ThreadStore()
