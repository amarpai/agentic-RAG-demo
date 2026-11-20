import math

import pytest

from payrag import metrics as M


def test_retrieval_metrics_known_values():
    retrieved, relevant = ["a", "b", "c", "d"], {"b", "d"}
    assert M.precision_at_k(retrieved, relevant, 2) == 0.5
    assert M.recall_at_k(retrieved, relevant, 2) == 0.5
    assert M.recall_at_k(retrieved, relevant, 4) == 1.0
    assert M.hit_at_k(retrieved, relevant, 1) == 0.0
    assert M.reciprocal_rank(retrieved, relevant) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert M.ndcg_at_k(retrieved, relevant, 4) == pytest.approx(dcg / idcg)
    assert M.reciprocal_rank(["x"], {"y"}) == 0.0


def test_confusion_matrix_and_report():
    labels = ["a", "b", "c"]
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    m = M.confusion_matrix(y_true, y_pred, labels)
    assert m == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    rep = M.classification_report(m, labels)
    assert rep["accuracy"] == pytest.approx(4 / 6)
    assert rep["per_label"]["b"]["precision"] == pytest.approx(2 / 3)
    assert rep["per_label"]["b"]["recall"] == 1.0
    assert rep["per_label"]["c"]["precision"] == 1.0
    assert rep["per_label"]["c"]["recall"] == 0.5
    f1s = [rep["per_label"][l]["f1"] for l in labels]
    assert rep["macro"]["f1"] == pytest.approx(sum(f1s) / 3)


def test_answer_metrics():
    assert M.fact_recall("Paid out T+2 at 23:00 UTC", ["t+2", "23:00 UTC", "T+1"]) == pytest.approx(2 / 3)
    assert M.fact_recall("anything", []) == 1.0
    assert M.citation_precision(["A", "B"], {"A"}) == 0.5
    assert M.citation_precision([], set()) == 1.0
    assert M.citation_precision([], {"A"}) == 0.0
    assert M.groundedness("cap is 21 cents", ["The cap is 21 cents plus 0.05%."]) == 1.0
    assert M.groundedness("cap is 99 dollars", ["The cap is 21 cents."]) == pytest.approx(1 / 3)
