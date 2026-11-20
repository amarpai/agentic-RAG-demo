"""Eval harness: run an agent over the golden set and score retrieval, routing and answers."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from payrag import metrics as M
from payrag.config import ALL_LABELS, GOLDEN_PATH, OUT_OF_SCOPE, THRESHOLDS_PATH

K_VALUES = (1, 3, 5)


@dataclass
class GoldenItem:
    id: str
    question: str
    category: str
    relevant_doc_ids: list[str]
    must_include: list[str]


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    with path.open() as f:
        return [GoldenItem(**json.loads(line)) for line in f if line.strip()]


def evaluate(agent, store, golden: list[GoldenItem]) -> dict:
    rows = []
    for item in golden:
        res = agent.run(item.question)
        relevant = set(item.relevant_doc_ids)
        in_scope = item.category != OUT_OF_SCOPE
        sources = [store.get_document(d).text for d in res.cited_doc_ids if store.get_document(d)]
        row = {
            "id": item.id, "question": item.question,
            "true_category": item.category, "pred_category": res.category,
            "relevant": item.relevant_doc_ids, "retrieved": res.retrieved_doc_ids[:10], "cited": res.cited_doc_ids,
            "answer": res.answer, "tool_calls": res.tool_calls, "latency_s": round(res.latency_s, 4),
            "error": res.error, "in_scope": in_scope,
        }
        if in_scope:
            for k in K_VALUES:
                row[f"recall@{k}"] = M.recall_at_k(res.retrieved_doc_ids, relevant, k)
                row[f"precision@{k}"] = M.precision_at_k(res.retrieved_doc_ids, relevant, k)
                row[f"hit@{k}"] = M.hit_at_k(res.retrieved_doc_ids, relevant, k)
            row["mrr"] = M.reciprocal_rank(res.retrieved_doc_ids, relevant)
            row["ndcg@5"] = M.ndcg_at_k(res.retrieved_doc_ids, relevant, 5)
            row["fact_recall"] = M.fact_recall(res.answer, item.must_include)
            row["citation_precision"] = M.citation_precision(res.cited_doc_ids, relevant)
            row["groundedness"] = M.groundedness(res.answer, sources)
        rows.append(row)
    return summarize(rows)


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows: list[dict]) -> dict:
    in_scope = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    y_true = [r["true_category"] for r in rows]
    y_pred = [r["pred_category"] for r in rows]
    matrix = M.confusion_matrix(y_true, y_pred, ALL_LABELS)
    report = M.classification_report(matrix, ALL_LABELS)

    retrieval = {key: _mean(in_scope, key)
                 for key in [*(f"{m}@{k}" for m in ("recall", "precision", "hit") for k in K_VALUES), "mrr", "ndcg@5"]}
    answer = {key: _mean(in_scope, key) for key in ("fact_recall", "citation_precision", "groundedness")}
    answer["exact_fact_match_rate"] = (sum(1 for r in in_scope if r["fact_recall"] == 1.0) / len(in_scope)) if in_scope else 0.0

    # Out-of-scope handling as a binary detector: positive = "should refuse".
    tp = sum(1 for r in oos if r["pred_category"] == OUT_OF_SCOPE)
    fp = sum(1 for r in in_scope if r["pred_category"] == OUT_OF_SCOPE)
    fn = len(oos) - tp
    abstention = {
        "refusal_recall": tp / len(oos) if oos else 1.0,
        "refusal_precision": tp / (tp + fp) if tp + fp else 1.0,
        "false_refusal_rate": fp / len(in_scope) if in_scope else 0.0,
        "binary_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": len(in_scope) - fp},
    }
    latencies = sorted(r["latency_s"] for r in rows)
    efficiency = {
        "mean_tool_calls": statistics.mean(r["tool_calls"] for r in rows) if rows else 0.0,
        "latency_p50_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_s": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0,
        "error_rate": sum(1 for r in rows if r["error"]) / len(rows) if rows else 0.0,
    }
    return {
        "n_questions": len(rows), "n_in_scope": len(in_scope), "n_out_of_scope": len(oos),
        "retrieval": retrieval,
        "routing": {"accuracy": report["accuracy"], "macro_f1": report["macro"]["f1"],
                    "weighted_f1": report["weighted"]["f1"], "per_label": report["per_label"]},
        "confusion_matrix": {"labels": ALL_LABELS, "matrix": matrix},
        "answer": answer, "abstention": abstention, "efficiency": efficiency,
        "rows": rows,
    }


# ----------------------------------------------------------------- thresholds
def flatten(summary: dict) -> dict[str, float]:
    flat = {}
    for section in ("retrieval", "answer", "abstention", "efficiency"):
        for k, v in summary[section].items():
            if isinstance(v, (int, float)):
                flat[f"{section}.{k}"] = v
    for k in ("accuracy", "macro_f1", "weighted_f1"):
        flat[f"routing.{k}"] = summary["routing"][k]
    return flat


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    """Return a list of human-readable failures. Keys ending in '_max' are upper bounds."""
    flat, failures = flatten(summary), []
    for key, bound in thresholds.items():
        metric = key[:-4] if key.endswith("_max") else key
        value = flat.get(metric)
        if value is None:
            failures.append(f"{metric}: metric not found")
        elif key.endswith("_max") and value > bound:
            failures.append(f"{metric} = {value:.3f} > max {bound}")
        elif not key.endswith("_max") and value < bound:
            failures.append(f"{metric} = {value:.3f} < min {bound}")
    return failures


def load_thresholds(agent_name: str, path: Path = THRESHOLDS_PATH) -> dict:
    return json.loads(path.read_text()).get(agent_name, {})


# -------------------------------------------------------------------- reports
def write_reports(summary: dict, out_dir: Path, agent_name: str, failures: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / agent_name
    (prefix.with_name(f"{agent_name}_results.json")).write_text(json.dumps(summary, indent=2) + "\n")

    labels, matrix = summary["confusion_matrix"]["labels"], summary["confusion_matrix"]["matrix"]
    with prefix.with_name(f"{agent_name}_confusion_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted", *labels])
        for label, row in zip(labels, matrix):
            w.writerow([label, *row])
    _plot_confusion(labels, matrix, prefix.with_name(f"{agent_name}_confusion_matrix.png"), agent_name)
    prefix.with_name(f"{agent_name}_report.md").write_text(render_markdown(summary, agent_name, failures))


def _plot_confusion(labels, matrix, path: Path, agent_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(matrix, cmap="Blues")
    short = [l.replace("_", "\n", 1) for l in labels]
    ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short, fontsize=8)
    peak = max(max(r) for r in matrix) or 1
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, v, ha="center", va="center", color="white" if v > peak / 2 else "black", fontsize=10)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(f"Intent routing confusion matrix ({agent_name} agent)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_markdown(s: dict, agent_name: str, failures: list[str]) -> str:
    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [f"# Eval report: `{agent_name}` agent", "",
             f"Golden set: {s['n_questions']} questions ({s['n_in_scope']} in scope, {s['n_out_of_scope']} out of scope).", "",
             f"**Threshold gate: {'PASS' if not failures else 'FAIL'}**", ""]
    lines += [f"- {f}" for f in failures]
    lines += ["", "## Retrieval", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in s["retrieval"].items()]
    lines += ["", "## Intent routing", "",
              f"Accuracy {pct(s['routing']['accuracy'])} · macro F1 {s['routing']['macro_f1']:.3f} · weighted F1 {s['routing']['weighted_f1']:.3f}", "",
              "| Label | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for label, m in s["routing"]["per_label"].items():
        lines.append(f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
    labels, matrix = s["confusion_matrix"]["labels"], s["confusion_matrix"]["matrix"]
    abbrev = [l[:6] for l in labels]
    lines += ["", "### Confusion matrix (rows = true, columns = predicted)", "",
              "| true \\ pred | " + " | ".join(abbrev) + " |", "|---" * (len(labels) + 1) + "|"]
    for label, row in zip(labels, matrix):
        lines.append(f"| {label} | " + " | ".join(str(v) if v else "·" for v in row) + " |")
    lines += ["", f"![confusion matrix]({agent_name}_confusion_matrix.png)", "",
              "## Answer quality", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in s["answer"].items()]
    a = s["abstention"]
    lines += ["", "## Out-of-scope abstention", "",
              f"Refusal recall {pct(a['refusal_recall'])} · refusal precision {pct(a['refusal_precision'])} · "
              f"false refusal rate {pct(a['false_refusal_rate'])} · {a['binary_confusion']}", "",
              "## Efficiency", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in s["efficiency"].items()]
    misses = [r for r in s["rows"] if r["pred_category"] != r["true_category"] or r.get("fact_recall", 1.0) < 1.0]
    lines += ["", "## Failures to inspect", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        lines.append(f"- **{r['id']}** {r['question']} — true `{r['true_category']}`, predicted `{r['pred_category']}`, "
                     f"fact recall {r.get('fact_recall', 1.0):.2f}, retrieved {r['retrieved'][:3]}")
    return "\n".join(lines) + "\n"
