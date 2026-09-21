"""
Renaming a conversation.

Titles are otherwise generated from the first question — a reasonable default and a poor label
for a thread someone returns to. The property that makes the feature real is that a chosen title
*stays* chosen: the interesting tests are the ones where automatic titling would previously have
overwritten it.
"""
import pytest

from app.core.database import db_client
from app.memory.conversation_memory import ConversationMemory
from app.memory.mongo_thread_store import thread_store


@pytest.fixture(autouse=True)
async def _connected():
    await db_client.connect()
    yield


async def _thread(api_client, cid, title="New Analysis"):
    return api_client.post("/api/conversations",
                           json={"repository_id": "repo_rename", "title": title}).json()


# ------------------------------------------------------------------ the basics

async def test_a_thread_can_be_renamed(api_client):
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]

    response = api_client.patch(f"/api/conversations/{cid}", json={"title": "Billing investigation"})
    assert response.status_code == 200
    assert response.json()["title"] == "Billing investigation"

    assert (await thread_store.read_meta(cid))["title"] == "Billing investigation"


async def test_the_new_title_shows_in_the_listing(api_client):
    """The sidebar reads the listing, so a rename that does not reach it is not a rename."""
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]
    api_client.patch(f"/api/conversations/{cid}", json={"title": "Payment flow"})

    listed = {t["conversation_id"]: t["title"] for t in api_client.get("/api/conversations").json()}
    assert listed[cid] == "Payment flow"


async def test_renaming_twice_keeps_the_latest(api_client):
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]
    api_client.patch(f"/api/conversations/{cid}", json={"title": "First"})
    api_client.patch(f"/api/conversations/{cid}", json={"title": "Second"})
    assert (await thread_store.read_meta(cid))["title"] == "Second"


# ------------------------------------------------------------------ the title must stick

async def test_a_renamed_thread_is_not_retitled_by_its_first_answer(api_client):
    """
    The case the old logic got wrong. Automatic titling fired whenever `message_count` was zero,
    so naming a brand-new thread and then asking a question silently undid the rename.
    """
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]
    api_client.patch(f"/api/conversations/{cid}", json={"title": "My debugging session"})

    await ConversationMemory.save_turn(cid, "Where is the discount calculation?",
                                       {"answer": "In DiscountService.", "sources": []})

    assert (await thread_store.read_meta(cid))["title"] == "My debugging session"


async def test_a_thread_renamed_to_the_default_string_still_sticks(api_client):
    """
    The other direction the magic-string test got wrong: choosing "New Analysis" deliberately
    made the thread eligible for automatic retitling again.
    """
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]
    api_client.patch(f"/api/conversations/{cid}", json={"title": "New Analysis"})

    await ConversationMemory.save_turn(cid, "What calls charge?", {"answer": "x", "sources": []})

    assert (await thread_store.read_meta(cid))["title"] == "New Analysis"


async def test_renaming_mid_conversation_holds(api_client):
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]

    await ConversationMemory.save_turn(cid, "first question", {"answer": "a", "sources": []})
    api_client.patch(f"/api/conversations/{cid}", json={"title": "Renamed later"})
    await ConversationMemory.save_turn(cid, "second question", {"answer": "b", "sources": []})

    assert (await thread_store.read_meta(cid))["title"] == "Renamed later"


async def test_an_untouched_thread_is_still_titled_automatically(api_client):
    """The default must keep working — renaming is an override, not a replacement."""
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]

    await ConversationMemory.save_turn(cid, "Where is the discount calculation?",
                                       {"answer": "x", "sources": []})

    assert (await thread_store.read_meta(cid))["title"] == "Where is the discount calculation?"


async def test_a_long_first_question_is_still_truncated(api_client):
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]
    question = "Explain in detail how the checkout pipeline validates and charges an order"

    await ConversationMemory.save_turn(cid, question, {"answer": "x", "sources": []})

    title = (await thread_store.read_meta(cid))["title"]
    assert title.endswith("...") and len(title) == 38


# ------------------------------------------------------------------ rejected input

async def test_an_empty_title_is_rejected(api_client):
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]

    for bad in ("", "   ", "\n\t"):
        response = api_client.patch(f"/api/conversations/{cid}", json={"title": bad})
        assert response.status_code == 400, f"accepted {bad!r}"

    assert (await thread_store.read_meta(cid))["title"] == "New Analysis", "a rejected rename still changed the title"


async def test_an_overlong_title_is_rejected(api_client):
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]
    response = api_client.patch(f"/api/conversations/{cid}", json={"title": "x" * 500})
    assert response.status_code == 400
    assert "120" in response.json()["detail"]


async def test_newlines_are_flattened_rather_than_refused(api_client):
    """A pasted multi-line title would break the sidebar row; it is cleaned, not rejected."""
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]

    response = api_client.patch(f"/api/conversations/{cid}",
                                json={"title": "  Billing\n\n   investigation\t "})
    assert response.status_code == 200
    assert response.json()["title"] == "Billing investigation"


async def test_renaming_an_unknown_thread_is_404(api_client):
    assert api_client.patch("/api/conversations/conv_nope", json={"title": "x"}).status_code == 404


async def test_a_rename_survives_a_reload(api_client):
    conv = await _thread(api_client, "c")
    cid = conv["conversation_id"]
    api_client.patch(f"/api/conversations/{cid}", json={"title": "Durable name"})

    body = api_client.get(f"/api/conversations/{cid}").json()
    assert body["title"] == "Durable name"
    assert body["title_is_custom"] is True
