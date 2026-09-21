"""
Per-thread folder storage and conversation resumption.

The thread folder holds the conversation and a `repository_id`. It does not hold the code index:
Neo4j and ChromaDB are per repository and shared across threads, so these tests assert both that
a thread round-trips from disk *and* that it routes to the right repository's graph.
"""
import json
import os

import pytest

from app.memory.thread_store import ThreadStore


@pytest.fixture
def store(tmp_path):
    return ThreadStore(str(tmp_path / "threads"))


# --------------------------------------------------------------------- folder lifecycle

def test_each_thread_gets_its_own_folder(store):
    for i in range(5):
        store.create(f"conv_{i}", "repo_shared", f"Thread {i}")

    folders = sorted(os.listdir(store.base_dir))
    assert folders == [f"conv_{i}" for i in range(5)]
    for folder in folders:
        assert os.path.isfile(os.path.join(store.base_dir, folder, "thread.json"))
        assert os.path.isfile(os.path.join(store.base_dir, folder, "messages.jsonl"))


def test_create_is_idempotent(store):
    first = store.create("conv_a", "repo_1", "Original")
    again = store.create("conv_a", "repo_1", "Different title")
    assert again["title"] == first["title"] == "Original"
    assert len(os.listdir(store.base_dir)) == 1


def test_thread_metadata_carries_the_repository_pointer(store):
    """The one field that routes to the index, the vector collection and the graph."""
    meta = store.create("conv_a", "repo_42", "t")
    assert meta["repository_id"] == "repo_42"
    on_disk = json.load(open(os.path.join(store.base_dir, "conv_a", "thread.json")))
    assert on_disk["repository_id"] == "repo_42"


# --------------------------------------------------------------------- messages

def test_messages_append_without_rewriting_history(store):
    store.create("conv_a", "repo_1")
    for turn in range(3):
        store.append_messages("conv_a", [
            {"role": "user", "content": f"q{turn}"},
            {"role": "assistant", "content": f"a{turn}"},
        ])

    messages = store.read_messages("conv_a")
    assert [m["content"] for m in messages] == ["q0", "a0", "q1", "a1", "q2", "a2"]
    assert store.read_meta("conv_a")["message_count"] == 6

    # One JSON object per line — the property that makes appending cheap.
    lines = open(store._messages_path("conv_a")).read().strip().split("\n")
    assert len(lines) == 6
    assert all(json.loads(line) for line in lines)


def test_message_limit_returns_the_most_recent(store):
    store.create("conv_a", "repo_1")
    store.append_messages("conv_a", [{"role": "user", "content": str(i)} for i in range(10)])
    assert [m["content"] for m in store.read_messages("conv_a", limit=3)] == ["7", "8", "9"]


def test_a_corrupt_line_does_not_destroy_the_thread(store):
    """A single JSON array would be unreadable end to end; JSONL loses only the bad line."""
    store.create("conv_a", "repo_1")
    store.append_messages("conv_a", [{"role": "user", "content": "before"}])
    with open(store._messages_path("conv_a"), "a") as handle:
        handle.write("{ not valid json\n")
    store.append_messages("conv_a", [{"role": "user", "content": "after"}])

    contents = [m["content"] for m in store.read_messages("conv_a")]
    assert contents == ["before", "after"]


def test_metadata_writes_are_atomic(store):
    store.create("conv_a", "repo_1")
    store.update_meta("conv_a", title="Updated")
    leftovers = [f for f in os.listdir(store.thread_dir("conv_a")) if f.endswith(".tmp")]
    assert not leftovers, f"interrupted-write temp files left behind: {leftovers}"


# --------------------------------------------------------------------- listing and deletion

def test_listing_filters_by_repository_and_sorts_newest_first(store):
    import time
    for conv_id, repo in [("c1", "repo_a"), ("c2", "repo_b"), ("c3", "repo_a")]:
        store.create(conv_id, repo)
        time.sleep(0.01)
        store.update_meta(conv_id, title=f"t-{conv_id}")

    assert {t["conversation_id"] for t in store.list_threads("repo_a")} == {"c1", "c3"}
    assert {t["conversation_id"] for t in store.list_threads("repo_b")} == {"c2"}
    assert len(store.list_threads()) == 3

    ordered = [t["conversation_id"] for t in store.list_threads()]
    assert ordered[0] == "c3", "newest first"


def test_listing_reads_metadata_only(store):
    """The sidebar must not pay for message history it does not show."""
    store.create("c1", "repo_a")
    store.append_messages("c1", [{"role": "user", "content": "x" * 10000}])
    listed = store.list_threads()[0]
    assert "messages" not in listed
    assert listed["message_count"] == 1


def test_delete_removes_only_that_thread(store):
    store.create("keep", "repo_1")
    store.create("drop", "repo_1")
    assert store.delete("drop") is True
    assert not os.path.exists(store.thread_dir("drop"))
    assert os.path.exists(store.thread_dir("keep"))
    assert store.delete("drop") is False


# --------------------------------------------------------------------- safety

@pytest.mark.parametrize("bad_id", ["../escape", "../../etc/passwd", "a/b", "./x"])
def test_thread_ids_cannot_escape_the_base_directory(store, bad_id):
    resolved = os.path.realpath(store.thread_dir(bad_id))
    assert resolved.startswith(os.path.realpath(store.base_dir))


@pytest.mark.parametrize("bad_id", ["..", "", "."])
def test_degenerate_thread_ids_are_rejected(store, bad_id):
    with pytest.raises(ValueError):
        store.thread_dir(bad_id)


def test_unknown_thread_returns_none_rather_than_raising(store):
    assert store.load("nope") is None
    assert store.read_meta("nope") is None
    assert store.read_messages("nope") == []


# --------------------------------------------------------------------- resumption

async def test_thread_resumes_and_routes_to_its_own_repository(temp_repo):
    """
    End to end: two threads on two repositories resolve the same follow-up to different
    symbols, using only the repository_id stored in each thread folder.
    """
    from app.memory.conversation_memory import ConversationMemory
    from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent
    from app.api.repositories import get_repository_chunks

    _, repo_a, _ = temp_repo({"billing.py": "def compute_invoice(i):\n    return apply_tax(sum(i))\n\ndef apply_tax(a):\n    return a * 1.2\n"})
    _, repo_b, _ = temp_repo({"shipping.py": "def estimate(o):\n    return pick_carrier(o)\n\ndef pick_carrier(o):\n    return 'dhl'\n"})

    for conv_id, repo_id, question in [("t_a", repo_a, "What depends on apply_tax?"),
                                       ("t_b", repo_b, "What depends on pick_carrier?")]:
        await ConversationMemory.get_or_create_conversation(conv_id, repo_id, "New Analysis")
        result = await Agent.process_user_query(
            repo_id, conv_id, question, get_repository_chunks(repo_id))
        await ConversationMemory.save_turn(conv_id, question, result)

    # Resume purely from stored state, as a restarted process would.
    for conv_id, repo_id, expected in [("t_a", repo_a, "compute_invoice"),
                                       ("t_b", repo_b, "estimate")]:
        thread = await ConversationMemory.load_conversation(conv_id)
        assert thread["repository_id"] == repo_id
        assert len(thread["messages"]) == 2

        result = await Agent.process_user_query(
            thread["repository_id"], conv_id, "Who calls it?",
            get_repository_chunks(thread["repository_id"]),
            working_context=thread["working_context"],
            history=thread["messages"],
        )
        assert expected in result["answer"], f"{conv_id} should resolve 'it' within its own repo"


async def test_threads_on_one_repository_keep_separate_context(temp_repo):
    from app.memory.conversation_memory import ConversationMemory
    from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent
    from app.api.repositories import get_repository_chunks

    _, repo_id, chunks = temp_repo({
        "billing.py": "def compute_invoice(i):\n    return apply_tax(sum(i))\n\ndef apply_tax(a):\n    return a * 1.2\n",
        "shipping.py": "def estimate(o):\n    return pick_carrier(o)\n\ndef pick_carrier(o):\n    return 'dhl'\n",
    })

    for conv_id, question in [("s1", "What depends on apply_tax?"),
                              ("s2", "What depends on pick_carrier?")]:
        await ConversationMemory.get_or_create_conversation(conv_id, repo_id, "New Analysis")
        result = await Agent.process_user_query(repo_id, conv_id, question, chunks)
        await ConversationMemory.save_turn(conv_id, question, result)

    first = (await ConversationMemory.load_conversation("s1"))["working_context"]["current_symbol"]
    second = (await ConversationMemory.load_conversation("s2"))["working_context"]["current_symbol"]
    assert first != second, "threads on one repository must not share a subject"


async def test_stored_turns_hold_pointers_not_code(temp_repo):
    """
    An assistant turn once persisted 306 KB of inlined source per message, of which the UI
    rendered about 1 KB. Citations are stored as (file, symbol, lines) and resolved on demand.
    """
    from app.memory.conversation_memory import ConversationMemory
    from app.agents.orchestrator import CodebaseAgentOrchestrator as Agent

    _, repo_id, chunks = temp_repo({"m.py": "def alpha():\n    return beta()\n\ndef beta():\n    return 1\n"})
    await ConversationMemory.get_or_create_conversation("p1", repo_id, "t")
    result = await Agent.process_user_query(repo_id, "p1", "How does alpha work?", chunks)
    await ConversationMemory.save_turn("p1", "How does alpha work?", result)

    messages = await ConversationMemory.load_conversation_history("p1")
    assistant = [m for m in messages if m["role"] == "assistant"][0]

    assert assistant["sources"], "citations should still be present"
    assert all(set(s) <= set(ConversationMemory.SOURCE_POINTER_FIELDS) for s in assistant["sources"])
    assert "code_snippet" not in json.dumps(assistant), "source bodies must not be persisted"
    assert set(assistant["execution_flow"]) <= {"feature", "flow_steps"}
