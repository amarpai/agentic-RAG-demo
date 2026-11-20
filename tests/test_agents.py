"""Agent behaviour: the offline baseline, and the Claude tool-use loop against a fake API client."""

from types import SimpleNamespace

import pytest

from payrag.agent import ClaudeResearchAgent, OfflineResearchAgent, build_agent


# ------------------------------------------------------------------- offline
def test_offline_answers_with_grounded_fact(store):
    res = OfflineResearchAgent(store).run("What is the daily clearing cut-off time?")
    assert res.category == "authorization_settlement"
    assert "23:00 UTC" in res.answer
    assert "AS-002" in res.cited_doc_ids
    assert set(res.cited_doc_ids) <= set(res.retrieved_doc_ids)
    assert res.tool_calls >= 3  # search, routed re-search, at least one document read


def test_offline_abstains_out_of_scope(store):
    res = OfflineResearchAgent(store).run("What is a good recipe for sourdough bread?")
    assert res.out_of_scope and res.cited_doc_ids == []


def test_build_agent_rejects_unknown(store):
    with pytest.raises(ValueError):
        build_agent("gpt", store)


# -------------------------------------------------------------------- claude
def _tool_use(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def _resp(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


class FakeClient:
    """Mimics client.beta.messages.create and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot the messages list, since the agent mutates it after the call.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_claude_loop_searches_reads_and_submits(store):
    client = FakeClient([
        _resp("tool_use",
              _tool_use("t1", "semantic_search", {"query": "representment deadline", "k": 3, "category": "any"}),
              _tool_use("t2", "get_document", {"doc_id": "CB-001"})),
        _resp("tool_use", _tool_use("t3", "submit_answer", {
            "answer": "Within 20 calendar days.", "category": "chargebacks_disputes",
            "cited_doc_ids": ["CB-001", "CB-001"]})),
    ])
    res = ClaudeResearchAgent(store, client=client).run("How long do merchants have to respond to a chargeback?")

    assert res.category == "chargebacks_disputes"
    assert res.cited_doc_ids == ["CB-001"]          # de-duplicated
    assert res.tool_calls == 3
    assert "CB-001" in res.retrieved_doc_ids and res.error is None

    # Request shape: model, adaptive thinking, effort and server-side fallback are set.
    first = client.requests[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["thinking"] == {"type": "adaptive"}
    assert first["fallbacks"] == "default"
    assert {t["name"] for t in first["tools"]} == {"semantic_search", "get_document", "submit_answer"}

    # Both tool results go back in ONE user message, after the unchanged assistant turn.
    second = client.requests[1]["messages"]
    assert second[1]["role"] == "assistant"
    results = second[2]["content"]
    assert second[2]["role"] == "user" and [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert not any(r.get("is_error") for r in results)


def test_claude_loop_reports_tool_errors_to_model(store):
    client = FakeClient([
        _resp("tool_use", _tool_use("t1", "get_document", {"doc_id": "NOPE-1"})),
        _resp("tool_use", _tool_use("t2", "submit_answer",
                                    {"answer": "n/a", "category": "out_of_scope", "cited_doc_ids": []})),
    ])
    res = ClaudeResearchAgent(store, client=client).run("q")
    err = client.requests[1]["messages"][2]["content"][0]
    assert err["is_error"] is True and "NOPE-1" in err["content"]
    assert res.out_of_scope


def test_claude_loop_nudges_then_gives_up_without_submit(store):
    text = SimpleNamespace(type="text", text="Some answer")
    client = FakeClient([_resp("end_turn", text), _resp("end_turn", text)])
    res = ClaudeResearchAgent(store, client=client).run("q")
    assert res.error == "no_submit_answer" and res.answer == "Some answer"
    assert "submit_answer" in client.requests[1]["messages"][-1]["content"]


def test_claude_loop_handles_refusal_and_step_limit(store):
    refused = ClaudeResearchAgent(store, client=FakeClient([_resp("refusal")])).run("q")
    assert refused.error == "refusal"

    looping = [_resp("tool_use", _tool_use(f"t{i}", "semantic_search", {"query": "x", "k": 1, "category": "any"}))
               for i in range(3)]
    res = ClaudeResearchAgent(store, client=FakeClient(looping), max_steps=3).run("q")
    assert res.error == "max_steps" and res.tool_calls == 3
