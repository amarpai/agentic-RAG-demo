"""Research agents.

Two interchangeable implementations share the `AgentResult` contract so the
same eval harness scores both:

* ClaudeResearchAgent  - an agentic tool-use loop on the Claude API. Claude
  plans searches, reformulates queries, filters by category, reads full
  documents, then calls `submit_answer` with a category and citations.
* OfflineResearchAgent - a deterministic multi-hop baseline (search -> route ->
  filtered re-search -> extractive answer). Needs no network or API key, which
  makes it suitable for CI regression tests.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from payrag.config import CLAUDE_EFFORT, CLAUDE_MODEL, OUT_OF_SCOPE
from payrag.tools import TOOLS, ToolExecutor
from payrag.vector_store import VectorStore


@dataclass
class AgentResult:
    question: str
    answer: str
    category: str
    cited_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    tool_calls: int
    latency_s: float
    trace: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def out_of_scope(self) -> bool:
        return self.category == OUT_OF_SCOPE


# ===================================================================== offline
_STOP = set("""a an the of for to in on at by is are was were be been it its this that with as or and
what which who how when where why do does did can could should would my our we i you your their there
under after before than then from into per any all each about tell me explain""".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower()) if t not in _STOP and len(t) > 1}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


class OfflineResearchAgent:
    """Deterministic agentic baseline: route, refine, extract."""

    name = "offline"

    def __init__(self, store: VectorStore, min_relevance: float = 0.27, semantic_weight: float = 0.25,
                 top_k: int = 5, answer_sentences: int = 2, fact_bonus: float = 1.5):
        self.store = store
        self.min_relevance = min_relevance
        self.semantic_weight = semantic_weight
        self.top_k = top_k
        self.answer_sentences = answer_sentences
        self.fact_bonus = fact_bonus
        vec = getattr(store, "vectorizer", None)
        self._idf_table = dict(zip(vec.get_feature_names_out(), vec.idf_)) if vec is not None else {}

    def _idf(self, term: str) -> float:
        return float(self._idf_table.get(term, 1.0))

    def run(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        tools = ToolExecutor(self.store)
        trace: list[dict] = []

        # Hop 1: broad search across the whole knowledge base.
        hits = tools.semantic_search(question, k=8)
        trace.append({"tool": "semantic_search", "input": {"query": question, "k": 8, "category": "any"},
                      "top": [h["doc_id"] for h in hits]})

        # Abstain when even the best match is weak: relevance = keyword cosine + w * semantic cosine.
        relevance = max((h["keyword_score"] + self.semantic_weight * h["semantic_score"] for h in hits), default=0.0)
        trace.append({"step": "relevance_gate", "relevance": round(relevance, 4), "min": self.min_relevance})
        if relevance < self.min_relevance:
            return AgentResult(question, "I could not find this in the NovaPay payments knowledge base.",
                               OUT_OF_SCOPE, [], tools.retrieved, 1, time.perf_counter() - t0, trace)

        # Route: score-weighted vote of the top hits decides the category.
        votes: dict[str, float] = defaultdict(float)
        for h in hits[:self.top_k]:
            votes[h["category"]] += h["score"]
        category = max(votes, key=votes.get)
        trace.append({"step": "route", "votes": dict(votes), "category": category})

        # Hop 2: refined search inside the chosen category.
        refined = tools.semantic_search(question, k=self.top_k, category=category)
        trace.append({"tool": "semantic_search", "input": {"query": question, "k": self.top_k, "category": category},
                      "top": [h["doc_id"] for h in refined]})
        # Retrieval order used for metrics: refined results first, then the rest of hop 1.
        ordered = [h["doc_id"] for h in refined] + [h["doc_id"] for h in hits]
        retrieved = list(dict.fromkeys(ordered))

        # Read the top 2 documents and extract the sentences that best match the question.
        # Sentence score = IDF-weighted term overlap, a bonus for concrete facts (numbers,
        # codes) since most questions ask for one, and a penalty for lower-ranked documents.
        q = _tokens(question)
        candidates = []
        for rank, doc_id in enumerate(retrieved[:2]):
            doc = tools.get_document(doc_id)
            trace.append({"tool": "get_document", "input": {"doc_id": doc_id}})
            for pos, sent in enumerate(_sentences(doc["text"])):
                overlap = sum(self._idf(t) for t in q & _tokens(sent))
                if overlap > 0 and re.search(r"\d", sent):
                    overlap += self.fact_bonus
                candidates.append((overlap - 1.0 * rank, -rank, -pos, sent, doc_id))
        candidates.sort(reverse=True)
        chosen = [c for c in candidates if c[0] > 0][: self.answer_sentences] or candidates[:1]
        answer = " ".join(c[3] for c in chosen)
        cited = list(dict.fromkeys(c[4] for c in chosen))
        return AgentResult(question, answer, category, cited, retrieved,
                           sum(1 for t in trace if "tool" in t), time.perf_counter() - t0, trace)


# ===================================================================== claude
SYSTEM_PROMPT = """You are a payments-industry research agent for NovaPay, a card and instant-payments processor.
Answer questions from merchants, risk analysts and engineers using ONLY the NovaPay knowledge base,
which you reach through the semantic_search and get_document tools.

How to work:
- Search first. If the top results are off-topic, reformulate the query (use payments terminology,
  synonyms, reason codes, message names) or apply a category filter, and search again.
- Open the most relevant documents with get_document before answering when snippets are not enough.
- Every fact in the answer must come from a document you cite. Quote numbers and thresholds exactly.
- Keep answers short: 1-3 sentences.
- Finish by calling submit_answer exactly once with the answer, the single best category, and the
  doc_ids you relied on.
- If the knowledge base does not cover the question (for example it is not about payments, or about a
  topic NovaPay has no documents on), call submit_answer with category "out_of_scope", an answer that
  says so, and an empty cited_doc_ids list. Do not answer from general knowledge."""


class ClaudeResearchAgent:
    """Agentic RAG loop on the Claude Messages API (manual tool-use loop)."""

    name = "claude"

    def __init__(self, store: VectorStore, client=None, model: str = CLAUDE_MODEL, effort: str = CLAUDE_EFFORT,
                 max_steps: int = 8):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.store = store
        self.model = model
        self.effort = effort
        self.max_steps = max_steps

    def _create(self, messages: list[dict]):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # On a safety-classifier refusal, the API re-runs the request on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def run(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        tools = ToolExecutor(self.store)
        messages: list[dict] = [{"role": "user", "content": question}]
        trace: list[dict] = []
        tool_calls = 0
        nudged = False

        def result(answer="", category=OUT_OF_SCOPE, cited=None, error=None):
            return AgentResult(question, answer, category, cited or [], tools.retrieved, tool_calls,
                               time.perf_counter() - t0, trace, error)

        for _ in range(self.max_steps):
            response = self._create(messages)
            if response.stop_reason == "refusal":
                return result(error="refusal")
            if response.stop_reason == "max_tokens":
                return result(error="max_tokens")

            # Always append the full assistant content (thinking blocks included) unchanged.
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if nudged:
                    text = " ".join(b.text for b in response.content if b.type == "text")
                    return result(answer=text, error="no_submit_answer")
                nudged = True
                messages.append({"role": "user", "content": "Please call submit_answer with your final answer."})
                continue

            results = []
            for block in tool_uses:
                tool_calls += 1
                trace.append({"tool": block.name, "input": block.input})
                if block.name == "submit_answer":
                    inp = block.input
                    return result(inp.get("answer", ""), inp.get("category", OUT_OF_SCOPE),
                                  list(dict.fromkeys(inp.get("cited_doc_ids", []))))
                try:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": tools.run(block.name, block.input)})
                except Exception as exc:  # report tool errors back to the model
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"Error: {exc}", "is_error": True})
            # All tool results go back in a single user message.
            messages.append({"role": "user", "content": results})

        return result(error="max_steps")


def build_agent(kind: str, store: VectorStore, **kwargs):
    if kind == "offline":
        return OfflineResearchAgent(store, **kwargs)
    if kind == "claude":
        return ClaudeResearchAgent(store, **kwargs)
    raise ValueError(f"Unknown agent '{kind}' (expected 'offline' or 'claude')")


def to_json(result: AgentResult) -> str:
    d = {k: v for k, v in result.__dict__.items() if k != "trace"}
    return json.dumps(d, indent=2)
