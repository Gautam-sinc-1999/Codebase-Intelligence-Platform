from typing import Dict, Any, List, Optional

from app.core.database import redis_client
from app.memory.mongo_thread_store import thread_store
from app.observability import current_trace_id


class ConversationMemory:
    """
    Thread persistence, written through to MongoDB on every turn (see `mongo_thread_store`).

    A thread holds the conversation and a `repository_id`. It does **not** hold the code index:
    Neo4j and ChromaDB are per repository and shared by every thread about that repository, so
    resuming a thread means reading its messages here and routing to the index by that one id.
    """

    # Citations are stored as pointers, not payloads.
    #
    # An assistant turn previously persisted `execution_flow.relevant_files` with the full source
    # of every matching file — 306 KB in one measured message, 98 % of the stored turn, of which
    # the UI renders `flow_steps` (about 1 KB) and nothing else. The snippets are already in the
    # index store and are re-readable by (file_path, start_line, end_line), so storing them again
    # per message bought nothing and grew without bound.
    SOURCE_POINTER_FIELDS = ("file_path", "symbol", "start_line", "end_line", "entity_type")

    @classmethod
    def _slim_sources(cls, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: s.get(k) for k in cls.SOURCE_POINTER_FIELDS} for s in (sources or [])]

    @classmethod
    def _slim_execution_flow(cls, flow: Dict[str, Any]) -> Dict[str, Any]:
        """Keeps what the UI renders; drops the per-file source bodies."""
        if not flow:
            return {}
        return {"feature": flow.get("feature", ""), "flow_steps": flow.get("flow_steps", [])}

    @classmethod
    def _slim_impact(cls, impact: Dict[str, Any]) -> Dict[str, Any]:
        if not impact:
            return {}
        return {
            "primary_target": impact.get("primary_target", {}),
            "confirmed_callers": impact.get("confirmed_callers", []),
            "confirmed_files": impact.get("confirmed_files", []),
            "affected_tests": impact.get("affected_tests", []),
            "inferred_impacts": impact.get("inferred_impacts", []),
        }

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    async def get_or_create_conversation(
        cls, conversation_id: str, repository_id: str, title: str = "New Analysis"
    ) -> Dict[str, Any]:
        meta = await thread_store.create(conversation_id, repository_id, title)
        await redis_client.set_state(conversation_id, meta.get("working_context", {}))
        return await thread_store.load(conversation_id) or meta

    @classmethod
    async def load_conversation(
        cls, conversation_id: str, message_limit: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        return await thread_store.load(conversation_id, message_limit=message_limit)

    @classmethod
    async def list_conversations(
        cls, repository_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return await thread_store.list_threads(repository_id=repository_id, limit=limit)

    @classmethod
    async def delete_conversation(cls, conversation_id: str) -> bool:
        return await thread_store.delete(conversation_id)

    # ------------------------------------------------------------------ turns

    @classmethod
    async def _repository_commit_sha(cls, repository_id: str) -> str:
        """
        The commit a repository is currently indexed at, or '' for one without an upstream.

        One small lookup per turn, which is nothing next to the LLM call it accompanies, and it
        is what lets a stored citation later be labelled as coming from an older version of the
        file rather than silently showing the wrong code.
        """
        if not repository_id:
            return ""
        try:
            from app.core.database import db_client
            record = await db_client.get_collection("repositories").find_one(
                {"repository_id": repository_id}
            )
            return (record or {}).get("commit_sha") or ""
        except Exception:
            return ""

    @classmethod
    async def save_turn(
        cls, conversation_id: str, user_message: str, assistant_response: Dict[str, Any]
    ) -> int:
        """
        Writes a question and its answer, and returns the answer's `seq`.

        Returned rather than discarded so the response can tell the client where the answer landed.
        The alternative is the client inferring it from its own array length, which is wrong the
        moment a thread is read back with a message limit.
        """
        from datetime import datetime
        now = datetime.utcnow().isoformat()

        meta = await thread_store.read_meta(conversation_id) or {}
        commit_sha = await cls._repository_commit_sha(meta.get("repository_id", ""))

        user_doc = {"role": "user", "content": user_message, "timestamp": now}
        assistant_doc = {
            "role": "assistant",
            "content": assistant_response.get("answer", ""),
            "intent": assistant_response.get("intent", ""),
            "sources": cls._slim_sources(assistant_response.get("sources", [])),
            "execution_flow": cls._slim_execution_flow(assistant_response.get("execution_flow", {})),
            "impact_analysis": cls._slim_impact(assistant_response.get("impact_analysis", {})),
            # Kept on the stored turn: a reader coming back to a thread should still be able to
            # see that a particular answer was assembled from a template, not written for them.
            "degraded": bool(assistant_response.get("degraded")),
            "timestamp": now,
        }
        # Stamped per message, not only on the thread: a long thread can straddle a sync, and the
        # turns before it describe different code from the turns after.
        if commit_sha:
            assistant_doc["commit_sha"] = commit_sha

        # Read from the ambient trace rather than passed in, so the buffered and the streaming
        # path both record it without either having to remember to. Feedback arrives minutes
        # after the trace has closed, and this id is the only way back to the retrieval that
        # produced the answer someone is rating.
        trace_id = current_trace_id()
        if trace_id:
            assistant_doc["trace_id"] = trace_id
        total = await thread_store.append_messages(conversation_id, [user_doc, assistant_doc])
        answer_seq = total - 1

        # The first question becomes the thread title, as it does in a chat sidebar — unless the
        # user has named it themselves, in which case their title stands for good.
        title = meta.get("title", "New Analysis")
        if not meta.get("title_is_custom"):
            if not meta.get("message_count") or title in ("New Analysis", "New Codebase Analysis"):
                title = user_message[:35] + ("..." if len(user_message) > 35 else "")

        updates: Dict[str, Any] = {"title": title}
        if commit_sha:
            updates["indexed_commit_sha"] = commit_sha
        working_context = assistant_response.get("working_context") or {}
        summary = assistant_response.get("updated_summary") or ""
        if working_context:
            updates["working_context"] = working_context
        if summary:
            updates["summary"] = summary
        await thread_store.update_meta(conversation_id, **updates)

        return answer_seq

        if working_context:
            await redis_client.set_state(conversation_id, working_context)

    @classmethod
    async def mark_repository_synced(
        cls, repository_id: str, from_sha: str, to_sha: str, files_changed: int
    ) -> int:
        """
        Records a version boundary in every thread about this repository.

        Applied to **all** of them, not only the thread that triggered the sync: every thread on
        the repository now cites code that has moved, so every one of them needs the boundary
        marked. Without it, the stale labels that start appearing on old citations look like a
        malfunction rather than the honest statement they are.

        Threads with no turns yet are skipped — there is nothing above the line to qualify.

        Returns the number of threads marked.
        """
        from datetime import datetime

        if not to_sha or from_sha == to_sha:
            return 0

        marked = 0
        for meta in await thread_store.list_threads(repository_id=repository_id, limit=1000):
            conversation_id = meta.get("conversation_id")
            if not conversation_id or not meta.get("message_count"):
                continue

            await thread_store.append_messages(conversation_id, [{
                "role": "system",
                "event": "repository_synced",
                "content": (
                    f"Synced to {to_sha[:7]} — {files_changed} file(s) changed. "
                    f"Answers above describe {from_sha[:7] or 'an earlier version'}."
                ),
                "from_commit": from_sha,
                "to_commit": to_sha,
                "files_changed": files_changed,
                "timestamp": datetime.utcnow().isoformat(),
            }])
            await thread_store.update_meta(conversation_id, indexed_commit_sha=to_sha)
            marked += 1

        return marked

    @classmethod
    async def load_conversation_history(cls, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        return await thread_store.read_messages(conversation_id, limit=limit)

    @classmethod
    def generate_incremental_summary(
        cls, existing_summary: str, user_msg: str, assistant_msg: str, working_ctx: Dict[str, Any]
    ) -> str:
        """Deterministic fallback retained for callers that have no LLM available."""
        from app.memory.summarizer import ConversationSummarizer
        return ConversationSummarizer.deterministic_summary(existing_summary, user_msg, working_ctx)
