"""Command line entry point: `payrag build-index | search | ask | eval`."""

from __future__ import annotations

import argparse
import sys

from payrag.config import REPORTS_DIR


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="payrag", description="Agentic RAG demo for the payments industry")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Generate the synthetic corpus and build the vector store")
    b.add_argument("--filler-per-category", type=int, default=10)

    s = sub.add_parser("search", help="Run a raw hybrid search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--category", default=None)

    a = sub.add_parser("ask", help="Ask the research agent a question")
    a.add_argument("question")
    a.add_argument("--agent", choices=["offline", "claude"], default="claude")
    a.add_argument("--trace", action="store_true", help="Print the tool-call trace")

    e = sub.add_parser("eval", help="Run the golden-set evals, write reports, and enforce thresholds")
    e.add_argument("--agent", choices=["offline", "claude"], default="offline")
    e.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions")
    e.add_argument("--no-gate", action="store_true", help="Do not exit non-zero when thresholds fail")

    args = p.parse_args(argv)

    from payrag.index import build_index, load_store

    if args.cmd == "build-index":
        store = build_index(filler_per_category=args.filler_per_category)
        print(f"Indexed {len(store.docs)} documents / {len(store.chunks)} chunks, dim={store.dense.shape[1]}")
        return 0

    store = load_store()
    if args.cmd == "search":
        for h in store.search(args.query, k=args.k, category=args.category):
            print(f"{h.score:.4f}  kw={h.sparse_score:.3f}  sem={h.dense_score:.3f}  [{h.category}] {h.doc_id}  {h.title}")
        return 0

    from payrag.agent import build_agent, to_json

    if args.cmd == "ask":
        result = build_agent(args.agent, store).run(args.question)
        print(to_json(result))
        if args.trace:
            for step in result.trace:
                print("  ·", step)
        return 0

    from payrag.evaluation import check_thresholds, evaluate, load_golden, load_thresholds, write_reports

    golden = load_golden()[: args.limit] if args.limit else load_golden()
    agent = build_agent(args.agent, store)
    summary = evaluate(agent, store, golden)
    failures = check_thresholds(summary, load_thresholds(args.agent)) if not args.limit else []
    write_reports(summary, REPORTS_DIR, args.agent, failures)

    r, rt, an, ab = summary["retrieval"], summary["routing"], summary["answer"], summary["abstention"]
    print(f"[{args.agent}] n={summary['n_questions']}  recall@5={r['recall@5']:.3f}  MRR={r['mrr']:.3f}  "
          f"nDCG@5={r['ndcg@5']:.3f}  routing_acc={rt['accuracy']:.3f}  macroF1={rt['macro_f1']:.3f}  "
          f"fact_recall={an['fact_recall']:.3f}  refusal_recall={ab['refusal_recall']:.3f}  "
          f"false_refusal={ab['false_refusal_rate']:.3f}")
    print(f"Reports written to {REPORTS_DIR}/{args.agent}_report.md")
    if failures:
        print("THRESHOLD FAILURES:\n  " + "\n  ".join(failures))
        return 0 if args.no_gate else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
