"""Build / load the knowledge base and vector index."""

from __future__ import annotations

from pathlib import Path

from payrag.config import CORPUS_PATH, INDEX_DIR, SEED
from payrag.data_gen import generate_corpus, read_corpus, write_corpus
from payrag.vector_store import LocalVectorStore


def build_index(corpus_path: Path = CORPUS_PATH, index_dir: Path = INDEX_DIR, seed: int = SEED,
                filler_per_category: int = 10) -> LocalVectorStore:
    docs = generate_corpus(seed=seed, filler_per_category=filler_per_category)
    write_corpus(docs, corpus_path)
    store = LocalVectorStore(docs)
    store.save(index_dir)
    return store


def load_store(corpus_path: Path = CORPUS_PATH, index_dir: Path = INDEX_DIR) -> LocalVectorStore:
    """Load the persisted index; rebuild it from the corpus if it is missing or unreadable."""
    try:
        return LocalVectorStore.load(index_dir)
    except Exception:
        if corpus_path.exists():
            store = LocalVectorStore(read_corpus(corpus_path))
            store.save(index_dir)
            return store
        return build_index(corpus_path, index_dir)
