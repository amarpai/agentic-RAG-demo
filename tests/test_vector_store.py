import numpy as np
import pytest

from payrag.vector_store import LocalVectorStore, chunk_documents


@pytest.mark.parametrize("query,expected", [
    ("pacs.008 customer credit transfer", "RTP-002"),
    ("Visa reason code 10.4 compelling evidence", "CB-002"),
    ("ISO 8583 0110 authorization response", "AS-001"),
    ("beneficial owner 25% ownership", "KYC-002"),
    ("Durbin Regulation II debit cap", "IP-002"),
])
def test_exact_term_queries_rank_the_right_doc_first(store, query, expected):
    assert store.search(query, k=3)[0].doc_id == expected


def test_paraphrased_query_finds_doc_in_top_5(store):
    hits = store.search("bots validating stolen card numbers with tiny payments", k=5)
    assert "FR-003" in [h.doc_id for h in hits]


def test_category_filter(store):
    hits = store.search("deadline", k=10, category="chargebacks_disputes")
    assert hits and all(h.category == "chargebacks_disputes" for h in hits)


def test_one_hit_per_document_and_sorted(store):
    hits = store.search("chargeback ratio remediation", k=10)
    assert len({h.doc_id for h in hits}) == len(hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_unrelated_query_has_no_keyword_match(store):
    hits = store.search("sourdough bread recipe", k=5)
    assert all(h.sparse_score == 0 for h in hits)


def test_embeddings_are_unit_norm(store):
    norms = np.linalg.norm(store.dense, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)
    assert store.embed(["instant payouts"]).shape == (1, store.dense.shape[1])


def test_save_and_load(tmp_path, store):
    store.save(tmp_path)
    for name in ("index.joblib", "embeddings.npy", "chunks.jsonl", "manifest.json"):
        assert (tmp_path / name).exists()
    loaded = LocalVectorStore.load(tmp_path)
    q = "card testing CAPTCHA"
    assert [h.doc_id for h in loaded.search(q)] == [h.doc_id for h in store.search(q)]


def test_chunking_splits_long_documents(corpus):
    doc = corpus[0]
    long_doc = type(doc)(**{**doc.__dict__, "doc_id": "LONG", "text": " ".join(["word"] * 300)})
    chunks = chunk_documents([long_doc], max_words=120, overlap=20)
    assert len(chunks) == 3 and all(c.doc_id == "LONG" for c in chunks)
