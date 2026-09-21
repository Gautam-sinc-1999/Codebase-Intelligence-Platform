"""
Moves folder-per-thread conversations into MongoDB, once.

Threads previously lived at `data/threads/<id>/{thread.json,messages.jsonl}`. They are the only
data in this system that cannot be regenerated, so the migration is deliberately conservative:

- **Nothing is deleted.** The folders stay exactly where they are. If the migration is wrong, or
  Mongo is later replaced, the original is still on disk to re-read.
- **A thread already in Mongo is skipped**, so running twice is harmless and a partially finished
  migration resumes rather than duplicating.
- **A thread that fails to migrate is logged and skipped**, not allowed to abort the rest. One
  unreadable folder must not block every other conversation from moving.
"""
import logging
from typing import Dict, Any

from app.memory.thread_store import thread_store as file_store
from app.memory.mongo_thread_store import thread_store as mongo_store

logger = logging.getLogger("memory.migration")


async def migrate_threads_to_mongo() -> Dict[str, Any]:
    """Copies every on-disk thread into MongoDB. Returns a summary of what happened."""
    outcome = {"found": 0, "migrated": [], "already_present": 0, "failed": []}

    try:
        on_disk = file_store.list_threads(limit=100000)
    except Exception as e:
        logger.warning("Could not read on-disk threads for migration: %s", e)
        return outcome

    outcome["found"] = len(on_disk)
    for meta in on_disk:
        conversation_id = meta.get("conversation_id")
        if not conversation_id:
            continue
        try:
            if await mongo_store.exists(conversation_id):
                outcome["already_present"] += 1
                continue

            await mongo_store.create(
                conversation_id,
                meta.get("repository_id", ""),
                meta.get("title", "New Analysis"),
            )
            # Carried over wholesale rather than field by field, so anything added to the
            # metadata since this was written comes across without touching this function.
            carried = {k: v for k, v in meta.items()
                       if k not in ("conversation_id", "message_count")}
            if carried:
                await mongo_store.update_meta(conversation_id, **carried)

            messages = file_store.read_messages(conversation_id)
            if messages:
                await mongo_store.append_messages(conversation_id, messages)

            outcome["migrated"].append(conversation_id)
        except Exception as e:
            logger.error("Could not migrate thread '%s': %s", conversation_id, e)
            outcome["failed"].append(conversation_id)

    if outcome["migrated"]:
        logger.info(
            "Migrated %d conversation(s) into MongoDB; the original folders were left in place.",
            len(outcome["migrated"]),
        )
    return outcome
