"""
Conversation threads in MongoDB, written through on every turn.

Threads are the only data in this system that cannot be regenerated — an index, its embeddings
and its graph all rebuild from source in seconds. So the properties worth testing are ordering,
durability and that a failure anywhere leaves a thread readable, not that it is fast.
"""
import pytest

from app.core.database import db_client
from app.memory.mongo_thread_store import thread_store, THREADS, MESSAGES
from app.memory.conversation_memory import ConversationMemory


@pytest.fixture(autouse=True)
async def _connected():
    await db_client.connect()
    yield


# ------------------------------------------------------------------ the basics

async def test_a_thread_round_trips(api_client):
    meta = await thread_store.create("conv_rt", "repo_1", "First thread")
    assert meta["conversation_id"] == "conv_rt"
    assert meta["repository_id"] == "repo_1"
    assert meta["message_count"] == 0

    assert await thread_store.exists("conv_rt")
    assert (await thread_store.read_meta("conv_rt"))["title"] == "First thread"


async def test_creating_twice_returns_the_existing_thread(api_client):
    first = await thread_store.create("conv_twice", "repo_1", "Original")
    second = await thread_store.create("conv_twice", "repo_1", "Different title")
    assert second["title"] == "Original", "a second create overwrote the thread"
    assert first["created_at"] == second["created_at"]


async def test_messages_come_back_in_the_order_they_were_written(api_client):
    """
    Ordered by an explicit `seq`, not a timestamp: two turns saved in the same millisecond must
    still come back in the order they happened.
    """
    await thread_store.create("conv_order", "repo_1")
    await thread_store.append_messages("conv_order", [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ])
    await thread_store.append_messages("conv_order", [{"role": "user", "content": "third"}])

    contents = [m["content"] for m in await thread_store.read_messages("conv_order")]
    assert contents == ["first", "second", "third"]


async def test_the_message_count_tracks_the_messages(api_client):
    await thread_store.create("conv_count", "repo_1")
    assert await thread_store.append_messages("conv_count", [{"role": "user", "content": "a"}]) == 1
    assert await thread_store.append_messages("conv_count", [
        {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]) == 3
    assert (await thread_store.read_meta("conv_count"))["message_count"] == 3


async def test_reading_the_last_n_messages(api_client):
    """The UI wants the recent turns, and a long thread must not be read in full to get them."""
    await thread_store.create("conv_limit", "repo_1")
    await thread_store.append_messages(
        "conv_limit", [{"role": "user", "content": f"m{i}"} for i in range(10)])

    recent = await thread_store.read_messages("conv_limit", limit=3)
    assert [m["content"] for m in recent] == ["m7", "m8", "m9"]


async def test_internal_fields_do_not_leak_into_the_api_shape(api_client):
    """
    `_id` and `conversation_id` are storage concerns, not part of a message.

    `seq` used to be on that list. It came off it when feedback arrived: a reader's rating has to
    name the answer it is about, and the client cannot use its array index to do so — a limited
    read (`GET /conversations/{id}?message_limit=N`) returns a *tail slice*, so on a long thread
    index 0 is not message 0. Addressing by index would have attached ratings to the wrong turn.
    """
    await thread_store.create("conv_clean", "repo_1")
    await thread_store.append_messages("conv_clean", [{"role": "user", "content": "hi"}])

    message = (await thread_store.read_messages("conv_clean"))[0]
    assert set(message) == {"role", "content", "seq"}
    assert message["seq"] == 0

    meta = await thread_store.read_meta("conv_clean")
    assert "_id" not in meta


async def test_a_limited_read_keeps_each_message_addressable(api_client):
    """
    The reason `seq` is exposed: on a tail slice, position in the array is not position in the
    thread. Feedback posted against an index would land on a different answer entirely.
    """
    await thread_store.create("conv_tail", "repo_1")
    await thread_store.append_messages("conv_tail", [
        {"role": "user", "content": f"m{n}"} for n in range(6)
    ])

    tail = await thread_store.read_messages("conv_tail", limit=2)
    assert [m["content"] for m in tail] == ["m4", "m5"]
    # Index 0 of the slice is message 4 of the thread — the whole point.
    assert [m["seq"] for m in tail] == [4, 5]


async def test_load_returns_metadata_and_messages_together(api_client):
    await thread_store.create("conv_load", "repo_1", "Loaded")
    await thread_store.append_messages("conv_load", [{"role": "user", "content": "q"}])

    thread = await thread_store.load("conv_load")
    assert thread["title"] == "Loaded"
    assert [m["content"] for m in thread["messages"]] == ["q"]


async def test_an_unknown_thread_is_none_rather_than_an_error(api_client):
    assert await thread_store.read_meta("conv_nope") is None
    assert await thread_store.load("conv_nope") is None
    assert await thread_store.read_messages("conv_nope") == []
    assert await thread_store.exists("conv_nope") is False


# ------------------------------------------------------------------ listing and deletion

async def test_threads_are_listed_newest_first_and_scoped_by_repository(api_client):
    import asyncio
    for i, repo in enumerate(["repo_a", "repo_a", "repo_b"]):
        await thread_store.create(f"conv_list_{i}", repo, f"Thread {i}")
        await asyncio.sleep(0.01)  # distinct updated_at

    listed = await thread_store.list_threads(repository_id="repo_a")
    ids = [t["conversation_id"] for t in listed]
    assert set(ids) == {"conv_list_0", "conv_list_1"}, "listing crossed repositories"
    assert ids[0] == "conv_list_1", "not newest first"

    everything = await thread_store.list_threads()
    assert len({t["conversation_id"] for t in everything}) >= 3


async def test_deleting_a_thread_removes_its_messages_too(api_client):
    """A thread's messages must not outlive it as unreachable rows."""
    await thread_store.create("conv_del", "repo_1")
    await thread_store.append_messages(
        "conv_del", [{"role": "user", "content": f"m{i}"} for i in range(4)])

    assert await thread_store.delete("conv_del") is True
    assert await thread_store.read_meta("conv_del") is None
    assert await thread_store.read_messages("conv_del") == []

    leftover = await db_client.get_collection(MESSAGES).find_one({"conversation_id": "conv_del"})
    assert leftover is None, "messages survived the thread that owned them"


async def test_deleting_is_idempotent(api_client):
    await thread_store.create("conv_gone", "repo_1")
    assert await thread_store.delete("conv_gone") is True
    assert await thread_store.delete("conv_gone") is False


async def test_deleting_one_thread_leaves_its_neighbour_intact(api_client):
    await thread_store.create("conv_keep", "repo_1")
    await thread_store.create("conv_drop", "repo_1")
    await thread_store.append_messages("conv_keep", [{"role": "user", "content": "keep me"}])
    await thread_store.append_messages("conv_drop", [{"role": "user", "content": "drop me"}])

    await thread_store.delete("conv_drop")

    assert [m["content"] for m in await thread_store.read_messages("conv_keep")] == ["keep me"]


# ------------------------------------------------------------------ write-through

async def test_a_turn_is_in_mongo_the_moment_it_is_saved(api_client):
    """
    The point of write-through rather than a checkpoint: there is no window in which a
    conversation exists only in a cache.
    """
    await ConversationMemory.get_or_create_conversation("conv_wt", "repo_1")
    await ConversationMemory.save_turn("conv_wt", "what does charge do?", {
        "answer": "It charges.",
        "sources": [{"file_path": "b.py", "symbol": "charge", "start_line": 1, "end_line": 2}],
    })

    # Read straight out of the collection, not through the store.
    rows = await db_client.get_collection(MESSAGES).find(
        {"conversation_id": "conv_wt"}).to_list(length=100)
    assert len(rows) == 2, "the turn was not durable immediately"
    assert {r["role"] for r in rows} == {"user", "assistant"}

    thread = await db_client.get_collection(THREADS).find_one({"conversation_id": "conv_wt"})
    assert thread["message_count"] == 2
    assert thread["title"] == "what does charge do?"


async def test_the_thread_survives_being_read_back_through_the_api(api_client):
    conv = api_client.post("/api/conversations",
                           json={"repository_id": "repo_1", "title": "api thread"}).json()
    cid = conv["conversation_id"]

    await ConversationMemory.save_turn(cid, "a question", {"answer": "an answer", "sources": []})

    reloaded = api_client.get(f"/api/conversations/{cid}").json()
    assert reloaded["message_count"] == 2
    assert [m["role"] for m in reloaded["messages"]] == ["user", "assistant"]

    listed = api_client.get("/api/conversations").json()
    assert cid in {t["conversation_id"] for t in listed}

    assert api_client.delete(f"/api/conversations/{cid}").status_code == 200
    assert api_client.get(f"/api/conversations/{cid}").status_code == 404
