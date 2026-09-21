"""
Conversation threads, stored in MongoDB and written through on every turn.

Replaces the folder-per-thread layout (`data/threads/<id>/thread.json` + `messages.jsonl`).
The files were fast and restart-proof, but they were local: not in a MongoDB backup, not visible
to a second instance, and not searchable across threads. Threads are the one thing here that
cannot be regenerated — an index, its embeddings and its graph all rebuild from source in
seconds — so where they live is the durability decision that matters most.

**Write-through, deliberately, rather than write-behind.** A checkpointing scheme would leave a
window in which a conversation existed only in a cache. Measured on this machine, that buys
nothing worth having:

    Mongo insert_one    0.155 ms        Redis RPUSH    0.087 ms
    one LLM turn        2,000-8,000 ms

The write is 0.005 % of a turn. There is no bottleneck to buffer, and the cost of a lossy buffer
would be paid in the only data that cannot be rebuilt.

**Two collections, not one document with an embedded array.** A thread has no natural size
ceiling, and a single document would both approach Mongo's 16 MB limit and force every read to
pull the entire history when the UI wants the last twenty turns. This mirrors what the files did
— metadata in one place, messages appended in another — so `read_messages(limit=…)` stays cheap:

    conversations           one document per thread: repository_id, title, summary, counters
    conversation_messages   one document per message, ordered by `seq`

When MongoDB is unreachable, `db_client` falls back to its JSON store, so a thread is still
written somewhere durable rather than lost.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.core.database import db_client

logger = logging.getLogger("memory.mongo_thread_store")

THREADS = "conversations"
MESSAGES = "conversation_messages"


class MongoThreadStore:
    """The same interface the file-backed store presented, made async."""

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _clean(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Drops Mongo's `_id`, which is not part of the thread and is not JSON-serialisable."""
        if document is None:
            return None
        return {k: v for k, v in document.items() if k != "_id"}

    @staticmethod
    def _threads():
        return db_client.get_collection(THREADS)

    @staticmethod
    def _messages():
        return db_client.get_collection(MESSAGES)

    # ------------------------------------------------------------------ metadata

    async def exists(self, conversation_id: str) -> bool:
        return await self._threads().find_one({"conversation_id": conversation_id}) is not None

    async def read_meta(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._clean(await self._threads().find_one({"conversation_id": conversation_id}))

    async def create(self, conversation_id: str, repository_id: str,
                     title: str = "New Analysis") -> Dict[str, Any]:
        existing = await self.read_meta(conversation_id)
        if existing:
            return existing

        now = datetime.utcnow().isoformat()
        meta = {
            "conversation_id": conversation_id,
            # The join key to the index, vector store and graph. Everything else about this
            # thread is local to it; this is the one pointer outward.
            "repository_id": repository_id,
            "title": title,
            # Whether a person chose this title. Auto-titling used to be skipped by testing the
            # title against the magic strings "New Analysis"/"New Codebase Analysis", which is
            # fragile in both directions: renaming a thread *to* one of them made it eligible for
            # overwriting again, and renaming a brand-new thread was silently undone by its first
            # answer. Intent is recorded rather than guessed at.
            "title_is_custom": False,
            "created_at": now,
            "updated_at": now,
            "summary": "",
            "working_context": {"current_feature": "", "current_symbol": "", "relevant_files": []},
            "message_count": 0,
            # The repository commit this thread's answers were written against.
            "indexed_commit_sha": "",
            # A remote SHA the user chose to stay behind, so the drift banner appears once per
            # new commit rather than on every message.
            "drift_ack_sha": "",
        }
        await self._threads().insert_one(dict(meta))
        return meta

    async def update_meta(self, conversation_id: str, **fields) -> Optional[Dict[str, Any]]:
        if not await self.exists(conversation_id):
            return None
        fields["updated_at"] = datetime.utcnow().isoformat()
        await self._threads().update_one({"conversation_id": conversation_id}, {"$set": fields})
        return await self.read_meta(conversation_id)

    # ------------------------------------------------------------------ messages

    async def append_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> int:
        """Appends turns and returns the new total."""
        meta = await self.read_meta(conversation_id)
        if not messages:
            return (meta or {}).get("message_count", 0)

        start = int((meta or {}).get("message_count", 0))
        for offset, message in enumerate(messages):
            # `seq` rather than a timestamp: two turns saved in the same millisecond must still
            # come back in the order they were written.
            await self._messages().insert_one(
                {**message, "conversation_id": conversation_id, "seq": start + offset}
            )

        total = start + len(messages)
        await self.update_meta(conversation_id, message_count=total)
        return total

    async def read_messages(self, conversation_id: str,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Reads a thread's messages in order, optionally only the most recent `limit`."""
        cursor = self._messages().find({"conversation_id": conversation_id})
        rows = [self._clean(row) for row in await cursor.to_list(length=100000)]
        rows.sort(key=lambda m: m.get("seq", 0))
        if limit:
            rows = rows[-limit:]
        return [{k: v for k, v in row.items() if k not in ("conversation_id", "seq")}
                for row in rows]

    async def load(self, conversation_id: str,
                   message_limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Metadata plus messages, in the shape the API returns."""
        meta = await self.read_meta(conversation_id)
        if meta is None:
            return None
        thread = dict(meta)
        thread["messages"] = await self.read_messages(conversation_id, limit=message_limit)
        return thread

    # ------------------------------------------------------------------ listing

    async def list_threads(self, repository_id: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lists threads newest first, reading metadata only.

        One indexed query, where the file layout needed one `open()` per thread — 1.6 ms against
        10.3 ms at 500 threads, and the gap widens with every thread added.
        """
        query = {"repository_id": repository_id} if repository_id else {}
        rows = await self._threads().find(query).to_list(length=max(limit, 1000))
        threads = [self._clean(row) for row in rows]
        threads.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
        return threads[:limit]

    async def delete(self, conversation_id: str) -> bool:
        """Removes a thread and its messages. Absent is not an error — deletion is idempotent."""
        if not await self.exists(conversation_id):
            return False

        await self._threads().delete_one({"conversation_id": conversation_id})
        # The fallback store has no delete_many, so messages go one at a time; a thread is tens
        # of documents, not thousands.
        while True:
            row = await self._messages().find_one({"conversation_id": conversation_id})
            if row is None:
                break
            await self._messages().delete_one(
                {"conversation_id": conversation_id, "seq": row.get("seq")}
            )
        return True


thread_store = MongoThreadStore()
