"""Tools the research agent can call, plus their JSON schemas for the Claude API."""

from __future__ import annotations

import json
from dataclasses import asdict

from payrag.config import ALL_LABELS, CATEGORIES
from payrag.vector_store import VectorStore

SEARCH_TOOL = {
    "name": "semantic_search",
    "description": (
        "Hybrid semantic + keyword search over NovaPay's payments knowledge base (policies, runbooks, "
        "reference docs, incident reports). Returns the top matching documents with a snippet and scores. "
        "Use focused queries; call it again with a reformulated query or a category filter if results look off-topic."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "k": {"type": "integer", "description": "Number of documents to return (1-10)."},
            "category": {"type": "string", "enum": ["any", *CATEGORIES],
                         "description": "Restrict results to one category, or 'any'."},
        },
        "required": ["query", "k", "category"],
        "additionalProperties": False,
    },
}

GET_DOC_TOOL = {
    "name": "get_document",
    "description": "Fetch the full text and metadata of one knowledge-base document by its doc_id (e.g. 'CB-001').",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"doc_id": {"type": "string"}},
        "required": ["doc_id"],
        "additionalProperties": False,
    },
}

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submit the final, grounded answer. Call exactly once, when research is done. "
        "Use category 'out_of_scope' with an empty citation list when the knowledge base cannot answer."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Concise answer using only facts from cited documents."},
            "category": {"type": "string", "enum": ALL_LABELS, "description": "Topic the question belongs to."},
            "cited_doc_ids": {"type": "array", "items": {"type": "string"},
                              "description": "doc_ids that support the answer."},
        },
        "required": ["answer", "category", "cited_doc_ids"],
        "additionalProperties": False,
    },
}

TOOLS = [SEARCH_TOOL, GET_DOC_TOOL, SUBMIT_TOOL]


class ToolExecutor:
    """Runs tool calls against a vector store and records which docs were retrieved."""

    def __init__(self, store: VectorStore):
        self.store = store
        self.retrieved: list[str] = []  # doc_ids in order of first retrieval

    def _note(self, doc_id: str) -> None:
        if doc_id not in self.retrieved:
            self.retrieved.append(doc_id)

    def semantic_search(self, query: str, k: int = 5, category: str = "any") -> list[dict]:
        k = max(1, min(int(k), 10))
        hits = self.store.search(query, k=k, category=None if category in (None, "any") else category)
        for h in hits:
            self._note(h.doc_id)
        return [{"doc_id": h.doc_id, "title": h.title, "category": h.category, "snippet": h.text[:400],
                 "score": h.score, "keyword_score": h.sparse_score, "semantic_score": h.dense_score} for h in hits]

    def get_document(self, doc_id: str) -> dict:
        doc = self.store.get_document(doc_id)
        if doc is None:
            raise KeyError(f"Unknown doc_id '{doc_id}'")
        self._note(doc.doc_id)
        return asdict(doc)

    def run(self, name: str, tool_input: dict) -> str:
        """Execute a tool and return its JSON-encoded result. Raises on bad input."""
        if name == "semantic_search":
            return json.dumps(self.semantic_search(tool_input["query"], tool_input.get("k", 5),
                                                   tool_input.get("category", "any")))
        if name == "get_document":
            return json.dumps(self.get_document(tool_input["doc_id"]))
        raise ValueError(f"Unknown tool '{name}'")
