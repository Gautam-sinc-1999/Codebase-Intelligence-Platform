"""
Reader feedback on an answer.

This is the only signal in the system that measures whether an answer was *useful*. Every other
score is computable from the answer and the index — whether its citations resolve, whether the
callers it named are real — and all of them can be perfect for an answer that did not help.

Two properties carry the feature:

1. **Feedback is kept even when tracing is off.** Langfuse is where it is analysed, not where it
   lives. Discarding what someone took the trouble to say, because a dashboard happened to be
   unconfigured, would be the worst possible failure mode for a feature like this.
2. **A rating lands on the answer it was about.** Ratings are addressed by `seq`, not by position
   in the returned array, because a limited read returns a tail slice.
"""
import pytest

from app.core.database import db_client
from app.memory.mongo_thread_store import thread_store
from app.observability import tracing


@pytest.fixture(autouse=True)
async def _connected():
    await db_client.connect()
    yield


async def _thread_with_answer(api_client, cid, trace_id="trace-abc"):
    """A thread holding one question and one answer, as a finished turn would leave it."""
    await thread_store.create(cid, "repo_feedback")
    assistant = {"role": "assistant", "content": "It is in billing.py.", "intent": "CODE_LOCATION"}
    if trace_id:
        assistant["trace_id"] = trace_id
    await thread_store.append_messages(cid, [
        {"role": "user", "content": "where is the total calculated?"},
        assistant,
    ])
    return cid


def _feedback(api_client, cid, seq, **body):
    return api_client.post(f"/api/conversations/{cid}/messages/{seq}/feedback", json=body)


# ------------------------------------------------------------------ recording a rating

async def test_a_thumbs_up_is_stored_on_the_turn(api_client):
    await _thread_with_answer(api_client, "fb_up")

    response = _feedback(api_client, "fb_up", 1, rating="up")
    assert response.status_code == 200
    assert response.json()["rating"] == "up"
    assert response.json()["stored"] is True

    message = await thread_store.read_message("fb_up", 1)
    assert message["feedback"] == "up"


async def test_a_thumbs_down_can_carry_a_comment(api_client):
    await _thread_with_answer(api_client, "fb_down")

    _feedback(api_client, "fb_down", 1, rating="down", comment="cited the wrong file")

    message = await thread_store.read_message("fb_down", 1)
    assert message["feedback"] == "down"
    assert message["feedback_comment"] == "cited the wrong file"


async def test_changing_your_mind_replaces_the_rating(api_client):
    """Last write wins — a reader who misclicks must be able to correct it."""
    await _thread_with_answer(api_client, "fb_change")

    _feedback(api_client, "fb_change", 1, rating="down", comment="wrong")
    _feedback(api_client, "fb_change", 1, rating="up")

    message = await thread_store.read_message("fb_change", 1)
    assert message["feedback"] == "up"
    # The old complaint must not survive the rating it belonged to.
    assert message["feedback_comment"] == ""


async def test_feedback_survives_a_reload(api_client):
    """The thumb has to still look pressed when the thread is reopened."""
    await _thread_with_answer(api_client, "fb_reload")
    _feedback(api_client, "fb_reload", 1, rating="up")

    thread = api_client.get("/api/conversations/fb_reload").json()
    answer = [m for m in thread["messages"] if m["role"] == "assistant"][0]
    assert answer["feedback"] == "up"


# ------------------------------------------------------------------ addressing the right answer

async def test_a_rating_lands_on_the_message_it_names_not_an_array_index(api_client):
    """
    The failure this guards: a limited read returns a tail slice, so the client's index 0 is not
    the thread's message 0. Ratings are addressed by `seq` for exactly this reason.
    """
    await thread_store.create("fb_long", "repo_feedback")
    await thread_store.append_messages("fb_long", [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "trace_id": "t1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "trace_id": "t2"},
    ])

    tail = api_client.get("/api/conversations/fb_long?message_limit=2").json()["messages"]
    answer = [m for m in tail if m["role"] == "assistant"][0]
    assert answer["content"] == "a2"

    _feedback(api_client, "fb_long", answer["seq"], rating="down")

    assert (await thread_store.read_message("fb_long", 3))["feedback"] == "down"
    assert "feedback" not in (await thread_store.read_message("fb_long", 1))


async def test_only_an_assistant_answer_can_be_rated(api_client):
    await _thread_with_answer(api_client, "fb_user")
    response = _feedback(api_client, "fb_user", 0, rating="up")
    assert response.status_code == 400


async def test_an_unknown_message_or_thread_is_a_404(api_client):
    await _thread_with_answer(api_client, "fb_404")
    assert _feedback(api_client, "fb_404", 99, rating="up").status_code == 404
    assert _feedback(api_client, "no_such_thread", 1, rating="up").status_code == 404


async def test_a_rating_must_be_up_or_down(api_client):
    await _thread_with_answer(api_client, "fb_bad")
    assert _feedback(api_client, "fb_bad", 1, rating="maybe").status_code == 400
    assert _feedback(api_client, "fb_bad", 1, rating="").status_code == 400


async def test_an_overlong_comment_is_refused(api_client):
    await _thread_with_answer(api_client, "fb_longcomment")
    response = _feedback(api_client, "fb_longcomment", 1, rating="down", comment="x" * 1001)
    assert response.status_code == 400


# ------------------------------------------------------------------ the Langfuse half

async def test_the_rating_is_scored_against_the_answers_own_trace(api_client, monkeypatch):
    """
    A thumb is only worth recording if it can be opened: the score has to land on the trace that
    produced the answer, so the retrieval behind a thumbs-down is one click away.
    """
    scored = []

    class _Client:
        def create_score(self, **kwargs):
            scored.append(kwargs)

    monkeypatch.setattr(tracing, "_client", _Client())
    monkeypatch.setattr(tracing, "_init_attempted", True)

    await _thread_with_answer(api_client, "fb_traced", trace_id="trace-xyz")
    response = _feedback(api_client, "fb_traced", 1, rating="down", comment="not useful")

    assert response.json()["traced"] is True
    assert len(scored) == 1
    assert scored[0]["trace_id"] == "trace-xyz"
    assert scored[0]["name"] == "user_feedback"
    assert scored[0]["value"] == 0.0
    assert scored[0]["comment"] == "not useful"


async def test_a_thumbs_up_scores_one(api_client, monkeypatch):
    """Up is 1 and down is 0, so the dashboard average reads as a satisfaction rate."""
    scored = []

    class _Client:
        def create_score(self, **kwargs):
            scored.append(kwargs)

    monkeypatch.setattr(tracing, "_client", _Client())
    monkeypatch.setattr(tracing, "_init_attempted", True)

    await _thread_with_answer(api_client, "fb_one")
    _feedback(api_client, "fb_one", 1, rating="up")

    assert scored[0]["value"] == 1.0


async def test_feedback_is_kept_when_tracing_is_off(api_client, monkeypatch):
    """
    The property that matters most: an unconfigured dashboard must not throw away what a reader
    said. `traced` reports the truth rather than the endpoint pretending it was recorded.
    """
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_init_attempted", True)

    await _thread_with_answer(api_client, "fb_untraced")
    response = _feedback(api_client, "fb_untraced", 1, rating="up")

    assert response.json() == {
        "conversation_id": "fb_untraced", "seq": 1, "rating": "up",
        "stored": True, "traced": False,
    }
    assert (await thread_store.read_message("fb_untraced", 1))["feedback"] == "up"


async def test_an_answer_written_without_a_trace_still_accepts_feedback(api_client, monkeypatch):
    """Answers produced while tracing was off have no trace id; the thumb must still work."""
    scored = []

    class _Client:
        def create_score(self, **kwargs):
            scored.append(kwargs)

    monkeypatch.setattr(tracing, "_client", _Client())
    monkeypatch.setattr(tracing, "_init_attempted", True)

    await _thread_with_answer(api_client, "fb_no_trace", trace_id=None)
    response = _feedback(api_client, "fb_no_trace", 1, rating="up")

    assert response.json()["stored"] is True
    assert response.json()["traced"] is False
    assert scored == []


async def test_a_failing_scorer_does_not_fail_the_request(api_client, monkeypatch):
    """Same rule as everywhere else: a broken dashboard degrades tracing, nothing else."""
    class _Exploding:
        def create_score(self, **_kwargs):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_client", _Exploding())
    monkeypatch.setattr(tracing, "_init_attempted", True)

    await _thread_with_answer(api_client, "fb_explode")
    response = _feedback(api_client, "fb_explode", 1, rating="up")

    assert response.status_code == 200
    assert response.json()["traced"] is False
    assert (await thread_store.read_message("fb_explode", 1))["feedback"] == "up"
