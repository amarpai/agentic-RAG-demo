"""Hybrid semantic vector store.

Each chunk gets two representations:

* a sparse TF-IDF vector (unigrams + bigrams) for exact-term matching
  (reason codes, ISO message names, thresholds), and
* a dense LSA embedding (TruncatedSVD over TF-IDF) that captures latent
  topical similarity between paraphrases.

Search fuses both rankings with Reciprocal Rank Fusion (RRF). Everything is
local and deterministic so the demo and its evals run with no network access.
To use a hosted vector DB instead, implement the `VectorStore` protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from payrag.data_gen import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    doc_type: str
    text: str


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    text: str
    score: float          # fused RRF score (higher is better)
    sparse_score: float   # cosine similarity in TF-IDF space, 0..1
    dense_score: float    # cosine similarity in LSA space, -1..1


class VectorStore(Protocol):
    def search(self, query: str, k: int = 5, category: str | None = None) -> list[SearchHit]: ...
    def get_document(self, doc_id: str) -> Document | None: ...


def chunk_documents(docs: list[Document], max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split documents into overlapping word windows; the title is prepended to every chunk."""
    chunks: list[Chunk] = []
    for d in docs:
        words = d.text.split()
        step = max_words - overlap
        starts = range(0, max(len(words) - overlap, 1), step) if len(words) > max_words else [0]
        for n, start in enumerate(starts):
            body = " ".join(words[start:start + max_words])
            chunks.append(Chunk(f"{d.doc_id}#{n}", d.doc_id, d.title, d.category, d.doc_type, f"{d.title}. {body}"))
    return chunks


class LocalVectorStore:
    RRF_K = 60
    MIN_DENSE = 0.2    # dense matches below this cosine are treated as noise
    DENSE_DEPTH = 30   # how many dense neighbours take part in fusion

    def __init__(self, docs: list[Document], n_components: int = 96, seed: int = 0):
        self.docs = {d.doc_id: d for d in docs}
        self.chunks = chunk_documents(docs)
        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english",
                                          token_pattern=r"(?u)\b\w+(?:\.\w+)*\b")
        self.sparse = self.vectorizer.fit_transform(texts)  # rows are L2-normalised
        n_components = min(n_components, self.sparse.shape[0] - 1, self.sparse.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.dense = normalize(self.svd.fit_transform(self.sparse))
        self._categories = np.array([c.category for c in self.chunks])

    # ------------------------------------------------------------- encode
    def embed(self, texts: list[str]) -> np.ndarray:
        """Dense embedding for arbitrary text (used for queries and exports)."""
        return normalize(self.svd.transform(self.vectorizer.transform(texts)))

    # ------------------------------------------------------------- search
    def search(self, query: str, k: int = 5, category: str | None = None) -> list[SearchHit]:
        q_sparse = self.vectorizer.transform([query])
        sparse_scores = (self.sparse @ q_sparse.T).toarray().ravel()
        dense_scores = self.dense @ self.embed([query])[0]

        mask = np.ones(len(self.chunks), dtype=bool) if category is None else (self._categories == category)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        # Each retriever only votes for chunks it actually matched; otherwise zero-score
        # chunks would earn RRF credit from their (arbitrary) tie order.
        rank_sparse = [i for i in idx[np.argsort(-sparse_scores[idx], kind="stable")] if sparse_scores[i] > 0]
        rank_dense = [i for i in idx[np.argsort(-dense_scores[idx], kind="stable")]
                      if dense_scores[i] >= self.MIN_DENSE][: self.DENSE_DEPTH]
        fused: dict[int, float] = {}
        for ranking in (rank_sparse, rank_dense):
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.RRF_K + r + 1)

        # Rank chunks, then de-duplicate to one hit per document.
        hits, seen = [], set()
        for i in sorted(fused, key=lambda j: (-fused[j], j)):
            c = self.chunks[i]
            if c.doc_id in seen:
                continue
            seen.add(c.doc_id)
            hits.append(SearchHit(c.chunk_id, c.doc_id, c.title, c.category, c.text,
                                  round(fused[i], 6), round(float(sparse_scores[i]), 4), round(float(dense_scores[i]), 4)))
            if len(hits) == k:
                break
        return hits

    def get_document(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)

    # -------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        """Persist the index plus a human-readable export of the vectors."""
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "index.joblib")
        np.save(directory / "embeddings.npy", self.dense.astype(np.float32))
        with (directory / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        meta = {"n_chunks": len(self.chunks), "n_docs": len(self.docs), "dim": int(self.dense.shape[1]),
                "vocab_size": len(self.vectorizer.vocabulary_), "fusion": f"RRF(k={self.RRF_K}) over TF-IDF + LSA"}
        (directory / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    @staticmethod
    def load(directory: Path) -> "LocalVectorStore":
        return joblib.load(directory / "index.joblib")
