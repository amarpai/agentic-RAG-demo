from collections import Counter

from payrag.config import CATEGORIES, GOLDEN_PATH
from payrag.data_gen import ANCHOR_DOCS, generate_corpus, read_corpus, write_corpus
from payrag.evaluation import load_golden


def test_corpus_is_deterministic():
    a, b = generate_corpus(seed=7), generate_corpus(seed=7)
    assert [d.text for d in a] == [d.text for d in b]
    assert [d.text for d in a] != [d.text for d in generate_corpus(seed=8)]


def test_doc_ids_unique_and_categories_balanced(corpus):
    ids = [d.doc_id for d in corpus]
    assert len(ids) == len(set(ids))
    counts = Counter(d.category for d in corpus)
    assert set(counts) == set(CATEGORIES)
    assert all(c == 14 for c in counts.values())  # 4 anchors + 10 fillers each
    assert len(ANCHOR_DOCS) == 32


def test_round_trip(tmp_path, corpus):
    path = tmp_path / "corpus.jsonl"
    write_corpus(corpus, path)
    assert read_corpus(path) == corpus


def test_golden_set_references_real_docs_and_facts(corpus):
    docs = {d.doc_id: d for d in corpus}
    golden = load_golden(GOLDEN_PATH)
    assert len(golden) >= 50
    for item in golden:
        assert item.category in CATEGORIES + ["out_of_scope"]
        for doc_id in item.relevant_doc_ids:
            assert doc_id in docs, item.id
            assert docs[doc_id].category == item.category, item.id
        # Every required fact must actually be stated in a relevant document.
        text = " ".join(docs[d].text.lower() for d in item.relevant_doc_ids)
        for fact in item.must_include:
            assert fact.lower() in text, (item.id, fact)
