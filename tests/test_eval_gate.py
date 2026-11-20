"""Measurable end-to-end quality gate: the offline agent must clear evals/thresholds.json."""

import pytest

from payrag.agent import OfflineResearchAgent
from payrag.config import ALL_LABELS
from payrag.evaluation import check_thresholds, evaluate, load_golden, load_thresholds


@pytest.fixture(scope="module")
def summary(store):
    return evaluate(OfflineResearchAgent(store), store, load_golden())


def test_offline_agent_meets_thresholds(summary):
    failures = check_thresholds(summary, load_thresholds("offline"))
    assert not failures, "\n".join(failures)


def test_confusion_matrix_is_consistent(summary):
    labels, matrix = summary["confusion_matrix"]["labels"], summary["confusion_matrix"]["matrix"]
    assert labels == ALL_LABELS
    assert sum(map(sum, matrix)) == summary["n_questions"]
    # Row sums equal the golden support per label.
    golden = load_golden()
    for i, label in enumerate(labels):
        assert sum(matrix[i]) == sum(1 for g in golden if g.category == label)
    diag = sum(matrix[i][i] for i in range(len(labels)))
    assert diag / summary["n_questions"] == pytest.approx(summary["routing"]["accuracy"])


def test_every_category_is_mostly_routed_correctly(summary):
    for label, m in summary["routing"]["per_label"].items():
        assert m["recall"] >= 0.5, f"{label} recall {m['recall']:.2f}"


def test_threshold_checker_detects_regressions(summary):
    assert check_thresholds(summary, {"retrieval.recall@5": 1.01}) == [
        f"retrieval.recall@5 = {summary['retrieval']['recall@5']:.3f} < min 1.01"]
    assert check_thresholds(summary, {"efficiency.mean_tool_calls_max": 0})[0].endswith("> max 0")
    assert check_thresholds(summary, {"nope.metric": 1}) == ["nope.metric: metric not found"]
