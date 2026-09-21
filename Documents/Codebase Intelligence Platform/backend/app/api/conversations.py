import uuid
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.core.database import db_client, sanitize_mongo_doc
from app.memory.conversation_memory import ConversationMemory
from app.agents.orchestrator import CodebaseAgentOrchestrator
from app.api.repositories import get_repository_chunks, indexing_status

router = APIRouter(prefix="/conversations", tags=["conversations"])

class CreateConversationRequest(BaseModel):
    repository_id: str
    title: Optional[str] = "New Analysis"

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    conversation_id: str
    repository_id: str
    intent: str
    answer: str
    sources: List[Dict[str, Any]]
    execution_flow: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    # True when the answer came from the rule-based template rather than a language model —
    # rate-limited, unreachable, or not configured. Defaulted so an older client is unaffected.
    degraded: bool = False
    degraded_reason: Optional[str] = None

@router.post("")
async def create_conversation(req: CreateConversationRequest):
    conv_id = f"conv_{uuid.uuid4().hex[:8]}"
    conv = await ConversationMemory.get_or_create_conversation(
        conversation_id=conv_id,
        repository_id=req.repository_id,
        title=req.title or "New Analysis"
    )
    return sanitize_mongo_doc(conv)

@router.get("")
async def list_conversations(repository_id: Optional[str] = None):
    """Thread list for the sidebar — metadata only, newest first, without loading any messages."""
    return await ConversationMemory.list_conversations(repository_id=repository_id)

@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, message_limit: Optional[int] = None):
    """
    Loads one thread for resumption: its messages, plus the `repository_id` the client uses to
    select the right repository — which is also what routes retrieval to the correct vector
    collection and graph on the next turn.
    """
    conversation = await ConversationMemory.load_conversation(conversation_id, message_limit)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    return conversation

async def _load_conversation_context(conversation_id: str):
    """
    Resolves a conversation to (conversation, repository_id, chunks), raising the appropriate
    HTTP error when it cannot be answered. Shared by the buffered and streaming endpoints so
    their validation cannot drift apart.
    """
    conv = await ConversationMemory.load_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation session not found")

    # The thread stores one pointer outward. Everything below — the chunk index, the vector
    # collection, the graph — is reached through it; none of it lives in the thread.
    repo_id = conv["repository_id"]

    # An empty chunk list is not a valid state to answer from. Defaulting to [] meant retrieval
    # found nothing and the response was assembled anyway, so "this repository is not indexed in
    # this process" was presented to the user as "I couldn't find that in your code" — a claim
    # about their codebase rather than about our state. Conversations now survive restarts
    # (F-13) while the chunk cache does not (F-15), which makes this reachable in normal use.
    all_chunks = get_repository_chunks(repo_id)
    if not all_chunks:
        repo = await db_client.get_collection("repositories").find_one({"repository_id": repo_id})
        if not repo:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Repository '{repo_id}' referenced by this conversation no longer exists. "
                    "Upload it again to continue this session."
                ),
            )
        state = indexing_status.get(repo_id, "not started")
        if state.startswith("failed"):
            detail = (
                f"Indexing failed for repository '{repo.get('name', repo_id)}' ({state}). "
                "Re-upload the repository to try again."
            )
        else:
            detail = (
                f"Repository '{repo.get('name', repo_id)}' is still being indexed (status: {state}). "
                "Retry in a moment."
            )
        raise HTTPException(status_code=409, detail=detail)

    return conv, repo_id, all_chunks


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(conversation_id: str, req: MessageRequest):
    conv, repo_id, all_chunks = await _load_conversation_context(conversation_id)

    # Process query with agent. Prior turns are passed so follow-up questions ("show me the
    # line numbers", "what about its callers") can resolve against what was already discussed.
    result = await CodebaseAgentOrchestrator.process_user_query(
        repository_id=repo_id,
        conversation_id=conversation_id,
        query=req.message,
        all_chunks=all_chunks,
        existing_summary=conv.get("summary", ""),
        working_context=conv.get("working_context", {}),
        history=conv.get("messages", [])
    )

    # Save turn directly to MongoDB conversation session document
    await ConversationMemory.save_turn(
        conversation_id=conversation_id,
        user_message=req.message,
        assistant_response=result
    )

    return MessageResponse(
        conversation_id=conversation_id,
        repository_id=repo_id,
        intent=result["intent"],
        answer=result["answer"],
        sources=result["sources"],
        execution_flow=result["execution_flow"],
        impact_analysis=result["impact_analysis"],
        degraded=result.get("degraded", False),
        degraded_reason=result.get("degraded_reason"),
    )

@router.post("/{conversation_id}/messages/stream")
async def stream_message(conversation_id: str, req: MessageRequest):
    """
    Same as POST /messages, but streams the answer as server-sent events.

    Events: `meta` (intent and sources, emitted before generation starts), then `token` deltas,
    then `done` carrying the full payload. Retrieval and analysis finish before the first token,
    so the UI can show sources and the execution flow while the prose is still arriving —
    previously the client waited on the entire turn, including a blocking LLM call, behind a
    spinner.
    """
    conv, repo_id, all_chunks = await _load_conversation_context(conversation_id)

    async def event_stream():
        final_result = None
        try:
            async for event in CodebaseAgentOrchestrator.stream_user_query(
                repository_id=repo_id,
                conversation_id=conversation_id,
                query=req.message,
                all_chunks=all_chunks,
                existing_summary=conv.get("summary", ""),
                working_context=conv.get("working_context", {}),
                history=conv.get("messages", []),
            ):
                if event["type"] == "done":
                    final_result = event["result"]
                    payload = {
                        "type": "done",
                        "intent": final_result["intent"],
                        "answer": final_result["answer"],
                        "degraded": final_result.get("degraded", False),
                        "degraded_reason": final_result.get("degraded_reason"),
                        "sources": final_result["sources"],
                        "execution_flow": final_result["execution_flow"],
                        "impact_analysis": final_result["impact_analysis"],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            # The response has already begun, so an error cannot become an HTTP status; it is
            # delivered as an event the client can render instead of a silently truncated answer.
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            return

        if final_result:
            # Persisted only after the stream completes, so an interrupted turn is not recorded
            # as a half-finished assistant message.
            await ConversationMemory.save_turn(
                conversation_id=conversation_id,
                user_message=req.message,
                assistant_response=final_result,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Removes the thread folder. The repository index is shared and deliberately untouched."""
    deleted = await ConversationMemory.delete_conversation(conversation_id)
    return {"status": "deleted" if deleted else "not found"}


class DriftAckRequest(BaseModel):
    sha: str


@router.post("/{conversation_id}/ack-drift")
async def acknowledge_drift(conversation_id: str, req: DriftAckRequest):
    """
    Records that the user chose to stay on the currently indexed version.

    Costs nothing — no clone, no re-index. Its only effect is to stop the drift banner
    reappearing for *this* commit: the acknowledgement is stored as the remote SHA that was
    declined, so the next commit upstream raises the question again rather than the banner being
    silenced forever.
    """
    from app.memory.mongo_thread_store import thread_store

    if not await thread_store.exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation session not found")

    sha = (req.sha or "").strip()
    if not sha:
        raise HTTPException(status_code=400, detail="A commit SHA is required.")

    await thread_store.update_meta(conversation_id, drift_ack_sha=sha)
    return {"conversation_id": conversation_id, "drift_ack_sha": sha, "status": "acknowledged"}


class RenameRequest(BaseModel):
    title: str


# Maximum length of a thread title. Long enough for a descriptive name, short enough that the
# sidebar row stays readable — auto-generated titles are already truncated to 35 characters.
MAX_TITLE_LENGTH = 120


@router.patch("/{conversation_id}")
async def rename_conversation(conversation_id: str, req: RenameRequest):
    """
    Renames a thread.

    Titles are otherwise generated from the first question, which is a reasonable default and a
    poor label for a conversation someone returns to. Renaming marks the title as deliberate, so
    the automatic titling never overwrites it afterwards.
    """
    from app.memory.mongo_thread_store import thread_store

    if not await thread_store.exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation session not found")

    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="A title cannot be empty.")
    if len(title) > MAX_TITLE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"A title cannot be longer than {MAX_TITLE_LENGTH} characters.",
        )
    # Newlines and control characters would break the sidebar row rather than being rejected
    # anywhere visible, so they are stripped instead of refused.
    title = " ".join(title.split())

    meta = await thread_store.update_meta(conversation_id, title=title, title_is_custom=True)
    return {"conversation_id": conversation_id, "title": meta["title"], "status": "renamed"}
