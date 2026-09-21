"""
Moving folder-per-thread conversations into MongoDB.

Threads cannot be regenerated, so the migration is conservative by design: it never deletes the
folders, it skips anything already migrated, and one bad thread does not stop the rest.
"""
import json
import os

import pytest

from app.core.database import db_client
from app.memory.thread_store import thread_store as file_store
from app.memory.mongo_thread_store import thread_store as mongo_store
from app.memory.thread_migration import migrate_threads_to_mongo


@pytest.fixture
def on_disk_thread():
    """A thread in the old layout, written directly through the file store."""
    def _make(conversation_id, repository_id="repo_1", turns=2, **meta):
        file_store.create(conversation_id, repository_id, "An older conversation")
        if meta:
            file_store.update_meta(conversation_id, **meta)
        file_store.append_messages(conversation_id, [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
            for i in range(turns)
        ])
        return conversation_id
    return _make


async def test_a_thread_on_disk_is_copied_into_mongo(api_client, on_disk_thread):
    on_disk_thread("conv_mig_1", turns=4)

    result = await migrate_threads_to_mongo()

    assert "conv_mig_1" in result["migrated"]
    assert await mongo_store.exists("conv_mig_1")
    assert [m["content"] for m in await mongo_store.read_messages("conv_mig_1")] == \
        ["turn 0", "turn 1", "turn 2", "turn 3"]


async def test_metadata_comes_across_whole(api_client, on_disk_thread):
    """
    Carried field by field would silently drop anything added since the migration was written.
    """
    on_disk_thread("conv_mig_meta", repository_id="repo_x",
                   summary="a summary", indexed_commit_sha="abc1234",
                   working_context={"current_symbol": "charge", "current_feature": "billing",
                                    "relevant_files": ["b.py"]})

    await migrate_threads_to_mongo()

    meta = await mongo_store.read_meta("conv_mig_meta")
    assert meta["repository_id"] == "repo_x"
    assert meta["summary"] == "a summary"
    assert meta["indexed_commit_sha"] == "abc1234"
    assert meta["working_context"]["current_symbol"] == "charge"
    assert meta["message_count"] == 2


async def test_the_folders_are_never_deleted(api_client, on_disk_thread):
    """
    A conversation is the only thing here that cannot be rebuilt. If the migration is wrong, the
    original has to still be there.
    """
    on_disk_thread("conv_mig_keep")
    folder = file_store.thread_dir("conv_mig_keep")

    await migrate_threads_to_mongo()

    assert os.path.isdir(folder)
    assert os.path.isfile(os.path.join(folder, "thread.json"))
    assert os.path.isfile(os.path.join(folder, "messages.jsonl"))


async def test_running_twice_does_not_duplicate_anything(api_client, on_disk_thread):
    on_disk_thread("conv_mig_twice", turns=3)

    first = await migrate_threads_to_mongo()
    second = await migrate_threads_to_mongo()

    assert "conv_mig_twice" in first["migrated"]
    assert "conv_mig_twice" not in second["migrated"]
    assert second["already_present"] >= 1
    assert len(await mongo_store.read_messages("conv_mig_twice")) == 3, "messages were duplicated"


async def test_a_thread_already_in_mongo_is_left_alone(api_client, on_disk_thread):
    """A partially finished migration must resume, not overwrite what already moved."""
    on_disk_thread("conv_mig_existing")
    await mongo_store.create("conv_mig_existing", "repo_1", "Already migrated")
    await mongo_store.append_messages("conv_mig_existing", [{"role": "user", "content": "newer"}])

    await migrate_threads_to_mongo()

    assert (await mongo_store.read_meta("conv_mig_existing"))["title"] == "Already migrated"
    assert [m["content"] for m in await mongo_store.read_messages("conv_mig_existing")] == ["newer"]


async def test_one_unreadable_thread_does_not_stop_the_others(api_client, on_disk_thread, monkeypatch):
    on_disk_thread("conv_mig_good_a")
    on_disk_thread("conv_mig_bad")
    on_disk_thread("conv_mig_good_b")

    real_read = file_store.read_messages

    def selective(conversation_id, limit=None):
        if conversation_id == "conv_mig_bad":
            raise OSError("this folder is corrupt")
        return real_read(conversation_id, limit)

    monkeypatch.setattr(file_store, "read_messages", selective)

    result = await migrate_threads_to_mongo()

    assert "conv_mig_bad" in result["failed"]
    assert "conv_mig_good_a" in result["migrated"]
    assert "conv_mig_good_b" in result["migrated"], "a later thread was skipped after a failure"


async def test_migrating_with_nothing_to_do_is_not_an_error(api_client, tmp_path):
    """
    Every startup runs this, and almost every startup will find nothing to move.

    Pointed at an empty directory rather than the session's, which earlier tests have populated —
    the original version of this test asserted "nothing on disk" while seven threads sat there,
    so it was measuring something other than its own name.
    """
    empty = tmp_path / "no_threads"
    empty.mkdir()
    original = file_store.base_dir
    file_store.base_dir = str(empty)
    try:
        result = await migrate_threads_to_mongo()
    finally:
        file_store.base_dir = original

    assert result["found"] == 0
    assert result["migrated"] == []
    assert result["failed"] == []


async def test_a_migrated_thread_is_usable_through_the_api(api_client, on_disk_thread):
    """The migration is only done if the thread answers the same way afterwards."""
    on_disk_thread("conv_mig_api", turns=2)
    await migrate_threads_to_mongo()

    response = api_client.get("/api/conversations/conv_mig_api")
    assert response.status_code == 200
    body = response.json()
    assert body["message_count"] == 2
    assert [m["content"] for m in body["messages"]] == ["turn 0", "turn 1"]

    assert api_client.delete("/api/conversations/conv_mig_api").status_code == 200
