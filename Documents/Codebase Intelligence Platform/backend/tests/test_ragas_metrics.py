"""
The RAGAS-style metrics and the judge that computes them.

Tested against a **recorded judge** — no network, no key, no cost. That is not only convenience:
the judge is the least reliable component in the system, so the properties worth pinning down are
what happens when it misbehaves. A metric that returns 0.0 because the judge timed out reports a
regression that did not happen, and the response to a regression is to revert something that was
fine.

So: every metric returns `None` when it cannot be computed, and never a number it did not measure.
"""
import json

import pytest

from app.observability import judge as judge_module
from app.observability import ragas_metrics as rm


class FakeJudge:
    """Replays canned replies in order, recording the prompts it was given."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []
        self.calls = 0

    async def ask(self, _client, prompt, **_kwargs):
        self.prompts.append(prompt)
        if not self._replies:
            return None
        self.calls += 1
        reply = self._replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)

    async def ask_json(self, client, prompt, **kwargs):
        return judge_module.parse_json(await self.ask(client, prompt, **kwargs))


@pytest.fixture(autouse=True)
def _judge_is_configured(monkeypatch):
    """Metrics short-circuit without a key; these tests are about what they do with one."""
    monkeypatch.setattr(judge_module.settings, "JUDGE_API_KEY", "test-key")


CONTEXTS = [
    "def calculate_discount(total):\n    return total * 0.1",
    "def process_checkout(cart):\n    return calculate_discount(cart.total)",
]


# ------------------------------------------------------------------ parsing a judge reply

def test_plain_json_is_parsed():
    assert judge_module.parse_json('{"a": 1}') == {"a": 1}


def test_a_fenced_block_is_parsed():
    """Models wrap JSON in fences however firmly they are told not to."""
    assert judge_module.parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_buried_in_prose_is_parsed():
    raw = 'Sure! Here is my assessment:\n{"a": 1}\nHope that helps.'
    assert judge_module.parse_json(raw) == {"a": 1}


def test_a_fence_is_preferred_over_a_brace_in_the_surrounding_prose():
    """
    The case the fenced branch exists for, and the only one that distinguishes it.

    Spanning the first `{` to the last `}` works right up until the prose *before* the fence
    contains a brace of its own — then that span swallows the fence markers and parses as
    nothing. Without this, the fenced branch looks redundant and a mutation removing it survives.
    """
    raw = 'Using the {rank} field:\n```json\n{"verdicts": [{"rank": 1}]}\n```\nDone.'
    assert judge_module.parse_json(raw) == {"verdicts": [{"rank": 1}]}


def test_an_unparseable_reply_is_none_rather_than_a_guess():
    assert judge_module.parse_json("I could not evaluate this.") is None
    assert judge_module.parse_json("") is None
    assert judge_module.parse_json(None) is None


# ------------------------------------------------------------------ faithfulness

async def test_faithfulness_is_the_supported_fraction_of_claims():
    judge = FakeJudge([{"claims": [
        {"claim": "it multiplies by 0.1", "supported": True},
        {"claim": "it also logs to Sentry", "supported": False},
        {"claim": "it returns a float", "supported": True},
        {"claim": "it retries on failure", "supported": False},
    ]}])

    got = await rm.faithfulness(judge, None, question="What does it do?",
                                answer="...", contexts=CONTEXTS)
    assert got["value"] == 0.5
    assert "2/4 claims supported" in got["comment"]
    assert got["detail"]["unsupported"] == 2


async def test_faithfulness_names_an_unsupported_claim():
    """Knowing the number dropped is useless without knowing which claim caused it."""
    judge = FakeJudge([{"claims": [
        {"claim": "it posts to Slack", "supported": False},
    ]}])
    got = await rm.faithfulness(judge, None, question="q", answer="a", contexts=CONTEXTS)
    assert got["value"] == 0.0
    assert "it posts to Slack" in got["comment"]


async def test_faithfulness_is_none_when_the_judge_fails():
    judge = FakeJudge([None])
    assert await rm.faithfulness(judge, None, question="q", answer="a",
                                 contexts=CONTEXTS) is None


async def test_faithfulness_is_none_with_nothing_to_judge():
    judge = FakeJudge([{"claims": [{"claim": "x", "supported": True}]}])
    assert await rm.faithfulness(judge, None, question="q", answer="",
                                 contexts=CONTEXTS) is None
    assert await rm.faithfulness(judge, None, question="q", answer="a", contexts=[]) is None
    # Neither case should have consulted the judge at all.
    assert judge.calls == 0


# ------------------------------------------------------------------ answer relevancy

def _fake_embedder(vectors_by_text):
    """Embeds by lookup, so a test can state the geometry it wants instead of mocking a model."""
    def embed(texts):
        return [vectors_by_text[t] for t in texts]
    return embed


async def test_answer_relevancy_is_the_mean_cosine_to_the_generated_questions():
    """
    RAGAS's actual method: generate questions from the answer, embed, compare to the original.

    The judge is never asked for a number — LLM-reported scores cluster on round values, so a
    real change of a few points cannot be told apart from the model rounding differently.
    """
    judge = FakeJudge([{"generated_questions": ["q1", "q2"], "noncommittal": False}])
    embed = _fake_embedder({
        "original": [1.0, 0.0],
        "q1": [1.0, 0.0],        # identical direction -> 1.0
        "q2": [0.0, 1.0],        # orthogonal          -> 0.0
    })

    got = await rm.answer_relevancy(judge, None, question="original", answer="a",
                                    embedder=embed)
    assert got["value"] == pytest.approx(0.5)
    assert got["detail"]["similarities"] == pytest.approx([1.0, 0.0])


async def test_a_perfectly_on_topic_answer_scores_one():
    judge = FakeJudge([{"generated_questions": ["same"], "noncommittal": False}])
    embed = _fake_embedder({"original": [0.6, 0.8], "same": [0.6, 0.8]})
    got = await rm.answer_relevancy(judge, None, question="original", answer="a", embedder=embed)
    assert got["value"] == pytest.approx(1.0)


async def test_a_noncommittal_answer_scores_zero_however_similar_it_looks():
    """
    "The context does not say which function handles this" is textually close to the question and
    answers nothing. Similarity alone would score it highly.
    """
    judge = FakeJudge([{"generated_questions": ["q1"], "noncommittal": True}])
    embed = _fake_embedder({"original": [1.0, 0.0], "q1": [1.0, 0.0]})

    got = await rm.answer_relevancy(judge, None, question="original", answer="I don't know",
                                    embedder=embed)
    assert got["value"] == 0.0
    assert "noncommittal" in got["comment"]


async def test_a_negative_cosine_is_floored_at_zero():
    """Cosine goes negative; a negative relevance is not a quantity worth averaging."""
    judge = FakeJudge([{"generated_questions": ["opposite"], "noncommittal": False}])
    embed = _fake_embedder({"original": [1.0, 0.0], "opposite": [-1.0, 0.0]})
    got = await rm.answer_relevancy(judge, None, question="original", answer="a", embedder=embed)
    assert got["value"] == 0.0


async def test_a_zero_vector_does_not_divide_by_zero():
    judge = FakeJudge([{"generated_questions": ["empty"], "noncommittal": False}])
    embed = _fake_embedder({"original": [0.0, 0.0], "empty": [1.0, 0.0]})
    got = await rm.answer_relevancy(judge, None, question="original", answer="a", embedder=embed)
    assert got["value"] == 0.0


async def test_answer_relevancy_is_none_when_no_questions_were_generated():
    judge = FakeJudge([{"generated_questions": [], "noncommittal": False}])
    assert await rm.answer_relevancy(judge, None, question="q", answer="a") is None


async def test_answer_relevancy_is_none_when_the_judge_says_nothing_usable():
    judge = FakeJudge(["not json at all"])
    assert await rm.answer_relevancy(judge, None, question="q", answer="a") is None


async def test_a_failing_embedder_yields_no_score_rather_than_zero():
    def explode(_texts):
        raise RuntimeError("onnx is unhappy")

    judge = FakeJudge([{"generated_questions": ["q1"], "noncommittal": False}])
    assert await rm.answer_relevancy(judge, None, question="q", answer="a",
                                     embedder=explode) is None


async def test_the_real_embedding_model_separates_on_topic_from_off_topic():
    """
    Exercises the **default** embedder, not an injected one.

    Every other test here passes its own `embedder`, which meant the default path — resolving and
    calling the vector store's model — was never run, and a wrong class name in it went unnoticed
    until the metric was tried for real. A metric whose only untested line is the one that loads
    the model is a metric that scores nothing in production.

    The thresholds are deliberately loose: the point is that the ordering is right and the scale
    is usable, not that a particular model returns a particular number.
    """
    question = "Where is the discount calculated?"
    on_topic = FakeJudge([{"noncommittal": False, "generated_questions": [
        "Which file contains the discount calculation?",
        "Where does the code compute a discount?",
    ]}])
    off_topic = FakeJudge([{"noncommittal": False, "generated_questions": [
        "What is the capital of France?",
        "How do I bake sourdough bread?",
    ]}])

    near = await rm.answer_relevancy(on_topic, None, question=question, answer="...")
    far = await rm.answer_relevancy(off_topic, None, question=question, answer="...")

    assert near["value"] > far["value"]
    assert near["value"] > 0.6, f"on-topic questions scored only {near['value']:.2f}"
    assert far["value"] < 0.4, f"unrelated questions scored {far['value']:.2f}"


async def test_the_judge_is_not_shown_the_original_question():
    """
    The generated questions must come from the answer alone. Showing the original invites the
    model to echo it back, which scores a perfect 1.0 for any answer at all.
    """
    judge = FakeJudge([{"generated_questions": ["q1"], "noncommittal": False}])
    embed = _fake_embedder({"Where is the retry logic?": [1.0, 0.0], "q1": [1.0, 0.0]})

    await rm.answer_relevancy(judge, None, question="Where is the retry logic?",
                              answer="It is in client.py.", embedder=embed)

    assert "Where is the retry logic?" not in judge.prompts[0]
    assert "It is in client.py." in judge.prompts[0]


# ------------------------------------------------------------------ context precision

async def test_context_precision_rewards_a_useful_chunk_ranked_first():
    judge = FakeJudge([{"verdicts": [{"rank": 1, "useful": True},
                                     {"rank": 2, "useful": False}]}])
    got = await rm.context_precision(judge, None, question="q", contexts=CONTEXTS)
    assert got["value"] == 1.0


async def test_context_precision_penalises_the_same_chunk_ranked_last():
    """
    The metric that scores the reranker. Retrieving the right chunk at position three is a worse
    run than retrieving it at position one, and an unweighted mean cannot tell them apart.
    """
    judge = FakeJudge([{"verdicts": [{"rank": 1, "useful": False},
                                     {"rank": 2, "useful": False},
                                     {"rank": 3, "useful": True}]}])
    got = await rm.context_precision(judge, None, question="q", contexts=CONTEXTS)
    assert got["value"] == pytest.approx(1 / 3)


async def test_context_precision_is_zero_when_nothing_retrieved_was_useful():
    judge = FakeJudge([{"verdicts": [{"rank": 1, "useful": False}]}])
    got = await rm.context_precision(judge, None, question="q", contexts=CONTEXTS)
    assert got["value"] == 0.0
    assert "no retrieved chunk" in got["comment"]


async def test_context_precision_is_none_without_context():
    judge = FakeJudge([{"verdicts": []}])
    assert await rm.context_precision(judge, None, question="q", contexts=[]) is None


# ------------------------------------------------------------------ entity recall

async def test_entity_recall_needs_no_judge():
    """
    The entities are symbol names the dataset already states. Asking a model to find them would
    add cost, latency and a chance of being wrong to a string search that is exact.
    """
    judge = FakeJudge([])
    got = await rm.context_entity_recall(
        judge, None, contexts=CONTEXTS,
        expected_entities=["calculate_discount", "process_checkout", "refund_order"],
    )
    assert got["value"] == pytest.approx(2 / 3)
    assert got["detail"]["missing"] == ["refund_order"]
    assert judge.calls == 0


async def test_entity_recall_does_not_count_a_symbol_inside_a_longer_name():
    """Same trap the intent classifier fell into — `_mentions` is word-boundary aware."""
    got = await rm.context_entity_recall(
        judge=FakeJudge([]), client=None,
        contexts=["def test_calculate_discount_rounding(): pass"],
        expected_entities=["calculate_discount"],
    )
    assert got["value"] == 0.0


async def test_entity_recall_is_none_without_ground_truth():
    assert await rm.context_entity_recall(FakeJudge([]), None, contexts=CONTEXTS,
                                          expected_entities=[]) is None


# ------------------------------------------------------------------ noise sensitivity

async def test_noise_sensitivity_is_the_corrupted_fraction_and_lower_is_better():
    judge = FakeJudge([{"total_claims": 4, "claims_from_irrelevant_context": 1,
                        "why": "took the retry loop from an unrelated chunk"}])
    got = await rm.noise_sensitivity(judge, None, question="q", answer="a", contexts=CONTEXTS)
    assert got["value"] == 0.25
    assert got["lower_is_better"] is True
    assert "retry loop" in got["comment"]


async def test_noise_sensitivity_is_none_when_the_judge_reports_no_claims():
    """Zero claims is not zero corruption; dividing by it would invent a perfect score."""
    judge = FakeJudge([{"total_claims": 0, "claims_from_irrelevant_context": 0}])
    assert await rm.noise_sensitivity(judge, None, question="q", answer="a",
                                      contexts=CONTEXTS) is None


# ------------------------------------------------------------------ the set

async def test_evaluate_answer_runs_every_metric_and_omits_what_it_cannot_measure():
    judge = FakeJudge([
        {"claims": [{"claim": "c", "supported": True}]},          # faithfulness
        {"generated_questions": ["g"], "noncommittal": False},    # answer_relevancy
        {"verdicts": [{"rank": 1, "useful": True}]},              # context_precision
        # context_entity_recall consults no judge
        None,                                                     # noise_sensitivity fails
    ])

    got = await rm.evaluate_answer(
        judge, None, question="q", answer="an answer", contexts=CONTEXTS,
        expected_entities=["calculate_discount"],
        embedder=_fake_embedder({"q": [1.0, 0.0], "g": [1.0, 0.0]}),
    )

    assert set(got) == {"faithfulness", "answer_relevancy", "context_precision",
                        "context_entity_recall"}
    assert "noise_sensitivity" not in got, "a failed metric must be omitted, not zeroed"


async def test_evaluate_answer_measures_nothing_without_a_judge_key(monkeypatch):
    monkeypatch.setattr(judge_module.settings, "JUDGE_API_KEY", "")
    got = await rm.evaluate_answer(FakeJudge([]), None, question="q", answer="a",
                                   contexts=CONTEXTS)
    assert got == {}


async def test_only_runs_the_named_metrics():
    """A full run is 15-30 judge calls; being able to run one metric matters on a free tier."""
    judge = FakeJudge([{"claims": [{"claim": "c", "supported": True}]}])
    got = await rm.evaluate_answer(judge, None, question="q", answer="a", contexts=CONTEXTS,
                                   only=["faithfulness"])
    assert set(got) == {"faithfulness"}
    assert judge.calls == 1


async def test_a_metric_that_raises_does_not_abandon_the_others():
    class Exploding(FakeJudge):
        async def ask_json(self, client, prompt, **kwargs):
            if "factual claims" in prompt:
                raise RuntimeError("judge exploded")
            return await super().ask_json(client, prompt, **kwargs)

    judge = Exploding([{"generated_questions": ["g"], "noncommittal": False}])
    got = await rm.evaluate_answer(judge, None, question="q", answer="a", contexts=CONTEXTS,
                                   only=["faithfulness", "answer_relevancy"],
                                   embedder=_fake_embedder({"q": [1.0, 0.0], "g": [1.0, 0.0]}))
    assert "faithfulness" not in got
    assert got["answer_relevancy"]["value"] == pytest.approx(1.0)


def test_the_metric_list_matches_what_is_implemented():
    """The names are pushed as scores; a name with no implementation is a silent gap."""
    assert set(rm.METRIC_NAMES) == set(rm.METRICS)
