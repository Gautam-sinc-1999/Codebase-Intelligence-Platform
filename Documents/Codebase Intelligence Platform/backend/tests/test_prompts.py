"""
Prompt management.

The prompt is not an optional part of a request. Tracing can be dropped when Langfuse is
unreachable and the user still gets their answer; a *prompt* cannot — without one the answer is
ungrounded, which is worse than no answer at all. So the property that matters most here is the
one that looks least interesting: **every failure path returns the in-code text.**

The second property is attribution. A version that produced an answer must be recorded on the
trace, or versioning buys rollback without ever telling you that something needed rolling back.
"""
import pytest

from app.observability import prompts
from app.observability import tracing


class FakeManagedPrompt:
    def __init__(self, text, version=7, labels=("production",), compiles=True):
        self.prompt = text
        self.version = version
        self.labels = list(labels)
        self.is_fallback = False
        self._text = text
        self._compiles = compiles

    def compile(self, **variables):
        if not self._compiles:
            raise ValueError("missing variable")
        text = self._text
        for key, value in variables.items():
            text = text.replace("{{" + key + "}}", str(value))
        return text


class FakeClient:
    def __init__(self, prompt=None, raises=False):
        self._prompt = prompt
        self._raises = raises
        self.requests = []

    def get_prompt(self, name, **kwargs):
        self.requests.append((name, kwargs))
        if self._raises:
            raise RuntimeError("langfuse is down")
        return self._prompt


@pytest.fixture
def langfuse_off(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_init_attempted", True)


def _use(monkeypatch, client):
    monkeypatch.setattr(tracing, "_client", client)
    monkeypatch.setattr(tracing, "_init_attempted", True)
    return client


# ------------------------------------------------------------------ the fallback guarantee

def test_the_in_code_text_is_used_when_langfuse_is_off(langfuse_off):
    resolved = prompts.get(prompts.ANSWER)
    assert resolved.text == prompts.ANSWER_PROMPT
    assert resolved.source == "fallback"
    assert resolved.is_managed is False
    assert resolved.client is None


def test_an_unreachable_langfuse_still_yields_a_prompt(monkeypatch):
    _use(monkeypatch, FakeClient(raises=True))
    resolved = prompts.get(prompts.ANSWER)
    assert resolved.text == prompts.ANSWER_PROMPT
    assert resolved.source == "fallback"


def test_a_prompt_that_will_not_compile_falls_back(monkeypatch):
    """A managed prompt expecting a variable nobody passed must not produce a broken prompt."""
    _use(monkeypatch, FakeClient(FakeManagedPrompt("hello {{name}}", compiles=False)))
    resolved = prompts.get(prompts.ANSWER)
    assert resolved.text == prompts.ANSWER_PROMPT
    assert resolved.source == "fallback"


def test_a_prompt_that_compiles_to_nothing_falls_back(monkeypatch):
    """An empty prompt is the worst outcome: the model answers with no grounding at all."""
    _use(monkeypatch, FakeClient(FakeManagedPrompt("   ")))
    resolved = prompts.get(prompts.ANSWER)
    assert resolved.text == prompts.ANSWER_PROMPT
    assert resolved.source == "fallback"


def test_the_sdks_own_fallback_is_not_reported_as_a_version(monkeypatch):
    """
    The SDK hands back our own text with `is_fallback` when it cannot reach Langfuse. Recording
    that as v7 would attribute a run to a version that was never actually used.
    """
    served = FakeManagedPrompt("whatever", version=7)
    served.is_fallback = True
    _use(monkeypatch, FakeClient(served))

    resolved = prompts.get(prompts.ANSWER)
    assert resolved.source == "fallback"
    assert resolved.version is None


def test_prompts_can_be_switched_off_without_losing_tracing(monkeypatch):
    """
    Turning off managed prompts must not require unsetting the Langfuse keys — that would take
    tracing down with it, which is the opposite of what someone pinning a prompt wants.
    """
    _use(monkeypatch, FakeClient(FakeManagedPrompt("a managed prompt")))
    monkeypatch.setattr(prompts.settings, "LANGFUSE_PROMPTS_ENABLED", "false")

    resolved = prompts.get(prompts.ANSWER)
    assert resolved.text == prompts.ANSWER_PROMPT
    assert resolved.source == "disabled"
    assert tracing.tracing_enabled() is True


# ------------------------------------------------------------------ the managed path

def test_a_managed_prompt_is_used_and_its_version_recorded(monkeypatch):
    client = _use(monkeypatch, FakeClient(FakeManagedPrompt("a reworded rule A1", version=4)))

    resolved = prompts.get(prompts.ANSWER)
    assert resolved.text == "a reworded rule A1"
    assert resolved.source == "langfuse"
    assert resolved.version == 4
    assert resolved.is_managed is True
    # The object itself travels to the generation, which is what creates the link.
    assert resolved.client is not None

    name, kwargs = client.requests[0]
    assert name == prompts.ANSWER
    assert kwargs["fallback"] == prompts.ANSWER_PROMPT, "the SDK needs our text for a cold cache"
    assert kwargs["cache_ttl_seconds"] > 0, "fetching per request would put Langfuse in the path"


def test_the_version_is_reported_for_the_trace(monkeypatch):
    _use(monkeypatch, FakeClient(FakeManagedPrompt("x", version=12)))
    assert prompts.get(prompts.ANSWER).describe() == {
        "prompt_name": "codebase-answer", "prompt_version": 12, "prompt_source": "langfuse",
    }


def test_a_fallback_describes_itself_honestly(langfuse_off):
    assert prompts.get(prompts.ANSWER).describe() == {
        "prompt_name": "codebase-answer", "prompt_version": None, "prompt_source": "fallback",
    }


def test_variables_are_substituted_on_both_paths(monkeypatch):
    """The fallback must fill placeholders too, or it renders `{{repo}}` to the model."""
    monkeypatch.setitem(prompts.REGISTRY, "tmp",
                        {"text": "about {{repo}}", "labels": [], "commit_message": "t"})

    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_init_attempted", True)
    assert prompts.get("tmp", repo="psf/requests").text == "about psf/requests"

    _use(monkeypatch, FakeClient(FakeManagedPrompt("managed {{repo}}")))
    assert prompts.get("tmp", repo="psf/requests").text == "managed psf/requests"


def test_an_unregistered_prompt_is_a_programming_error():
    """Silently returning nothing would send an empty system prompt to the model."""
    with pytest.raises(KeyError):
        prompts.get("no-such-prompt")


# ------------------------------------------------------------------ wiring

def test_the_summariser_uses_the_registered_text():
    from app.memory.summarizer import ConversationSummarizer

    assert ConversationSummarizer.SYSTEM_PROMPT == prompts.SUMMARY_PROMPT


def test_both_prompts_are_registered_with_commit_messages():
    assert set(prompts.REGISTRY) == {prompts.ANSWER, prompts.SUMMARY}
    for name, spec in prompts.REGISTRY.items():
        assert spec["text"].strip(), f"{name} has no text"
        assert spec["commit_message"], f"{name} has no commit message to explain the version"


def test_the_answering_prompt_still_carries_the_grounding_rules():
    """
    These are the rules the system's correctness rests on — A1 in particular exists because the
    model averaged the complete graph facts against the partial snippets. A refactor that moved
    the text must not have dropped any of it.
    """
    text = prompts.ANSWER_PROMPT
    assert "AUTHORITATIVE and COMPLETE" in text
    assert "Never describe a list given in the facts as partial" in text
    assert "Never invent file paths" in text
    assert len(text) == 1739, "the prompt changed; update the budget figures in the docs too"
