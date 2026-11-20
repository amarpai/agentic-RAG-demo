"""Pure-python evaluation metrics: retrieval, classification and answer quality."""

from __future__ import annotations

import math
import re


# ------------------------------------------------------------------ retrieval
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


# -------------------------------------------------------------- classification
def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    """Rows = true label, columns = predicted label. Unknown predictions are ignored."""
    index = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            m[index[t]][index[p]] += 1
    return m


def classification_report(matrix: list[list[int]], labels: list[str]) -> dict:
    """Per-label precision / recall / F1 / support plus accuracy and macro / weighted averages."""
    n = len(labels)
    total = sum(map(sum, matrix))
    per_label = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n)) - tp
        fn = sum(matrix[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_label[label] = {"precision": p, "recall": r, "f1": f1, "support": sum(matrix[i])}
    supported = [v for v in per_label.values() if v["support"]]
    accuracy = sum(matrix[i][i] for i in range(n)) / total if total else 0.0
    macro = {k: sum(v[k] for v in supported) / len(supported) if supported else 0.0
             for k in ("precision", "recall", "f1")}
    weighted = {k: sum(v[k] * v["support"] for v in supported) / total if total else 0.0
                for k in ("precision", "recall", "f1")}
    return {"per_label": per_label, "accuracy": accuracy, "macro": macro, "weighted": weighted, "n": total}


# -------------------------------------------------------------- answer quality
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def fact_recall(answer: str, must_include: list[str]) -> float:
    """Share of required facts (case-insensitive substrings) present in the answer."""
    if not must_include:
        return 1.0
    a = _norm(answer)
    return sum(1 for f in must_include if _norm(f) in a) / len(must_include)


def citation_precision(cited: list[str], relevant: set[str]) -> float:
    """Share of cited documents that are actually relevant. 1.0 when nothing should be cited and nothing is."""
    if not cited:
        return 1.0 if not relevant else 0.0
    return sum(1 for d in cited if d in relevant) / len(cited)


def _content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\$?[a-z0-9]+(?:[.,][a-z0-9]+)*%?", text.lower()) if len(t) > 2 or any(ch.isdigit() for ch in t)]


def groundedness(answer: str, source_texts: list[str]) -> float:
    """Share of answer content tokens that appear in the cited sources (a cheap faithfulness proxy)."""
    tokens = _content_tokens(answer)
    if not tokens:
        return 0.0
    vocab = set(_content_tokens(" ".join(source_texts)))
    return sum(1 for t in tokens if t in vocab) / len(tokens)
