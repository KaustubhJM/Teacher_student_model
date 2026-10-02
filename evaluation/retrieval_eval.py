"""
evaluation/retrieval_eval.py — score the doubt retriever (tools.search_document)
against a golden dataset with the LLM-judged recall / precision metrics.

Baseline = exactly what search_document does for the golden question:
    retrieve_with_scores_indexed(index, q, k=3)
    -> expand_with_neighbors(window=1, min_score=0.3)
    -> sorted best-first (since 2026-09-29; page order before)
    -> reranker filter rr >= -8, reranker order (since 2026-10-02)
Every question is also run through search_document itself and any
difference is reported, so the baseline can't silently drift from tools.py.

Precision is reported for two orders of the same chunks: the order the LLM
actually receives ("sent order" — best-first now, page order in results
before 2026-09-29) and score order (best first). Recall doesn't depend on
order.

--sweep: recall vs precision tradeoff. The same questions under other
retrieval settings (see SWEEP): top-k with no expansion (the pure ranking
curve), neighbour window 0/1/2 at k=3, a cut on direct-hit similarity, and
a cross-encoder reranker (score threshold + cap; candidate proposal only,
not in tools.py). Only chunks no earlier run judged cost a judge call.

Run:
    python -m evaluation.retrieval_eval golden/dbms_u5.jsonl
    python -m evaluation.retrieval_eval golden/dbms_u5.jsonl --limit 2      # dry run
    python -m evaluation.retrieval_eval golden/dbms_u5.jsonl --sweep
    python -m evaluation.retrieval_eval golden/tc_u5.jsonl --sweep --core   # skip wide settings
Writes evaluation/results/retrieval_baseline_<timestamp>.json and .md
(retrieval_sweep_<timestamp>.* with --sweep).
Judge results are cached, so a rerun (e.g. after a Groq daily limit) only
judges what is missing.
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

from deepeval.test_case import LLMTestCase

from .judge import JUDGE_MODEL
from .retrieval_metrics import (ContextualClaimRecallMetric, RankedContextualPrecisionMetric,
                                RetrievalJudge, weighted_cumulative_precision)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REQUIRED_FIELDS = ("id", "source", "question", "ideal_answer", "in_document")

# Must match tools.search_document (checked at run time against the real tool).
BASELINE = {"k": 3, "window": 1, "neighbor_min_score": 0.3}

# Tradeoff sweep. direct_min = similarity cut on direct hits (search_document
# has none today); nbr_min = expand_with_neighbors' min_score.
# "wide" settings add many chunks to judge (on a 28-chunk index ~12 per
# question vs ~7); --core skips them — they were settled on DBMS.
SWEEP = (
    [{"name": f"top-{k}", "group": "k (no expansion)", "k": k, "window": 0, "nbr_min": None,
      "direct_min": 0.0, "wide": k >= 6} for k in (1, 2, 3, 4, 5, 6, 8, 10)]
    + [{"name": f"k3 window {w}" + (" (no reranker)" if w == 1 else ""), "group": "neighbour window",
        "k": 3, "window": w, "nbr_min": 0.3, "direct_min": 0.0, "wide": w == 2} for w in (0, 1, 2)]
    + [{"name": f"k3 w1 direct-min {m}", "group": "direct-hit min score", "k": 3, "window": 1,
        "nbr_min": 0.3, "direct_min": m} for m in (0.2, 0.3, 0.4)]
    + [{"name": "k4 window 1", "group": "k=4 + neighbour window", "k": 4, "window": 1,
        "nbr_min": 0.3, "direct_min": 0.0, "wide": True}]
    # Cross-encoder reranker: score (question, chunk) pairs, keep score >=
    # min_score (best first), at most max_chunks. Threshold -2.1 was picked on
    # the DBMS set (2026-09-29) — the TC set is its held-out check.
    + [{"name": "rerank top-10, rr>=-2.1, max 8", "group": "cross-encoder reranker", "k": 10,
        "window": 0, "nbr_min": None, "direct_min": 0.0,
        "rerank": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "min_score": -2.1, "max_chunks": 8}},
       {"name": "k3 w1 chunks, rr>=-2.1", "group": "cross-encoder reranker", "k": 3,
        "window": 1, "nbr_min": 0.3, "direct_min": 0.0,
        "rerank": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "min_score": -2.1, "max_chunks": None}},
       {"name": "k3 w1 chunks, rr>=-8 (current tool)", "group": "cross-encoder reranker", "k": 3,
        "window": 1, "nbr_min": 0.3, "direct_min": 0.0,
        "rerank": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "min_score": -8.0, "max_chunks": None}}]
)

_rerankers: Dict[str, object] = {}


def _get_reranker(model: str):
    if model not in _rerankers:
        from sentence_transformers import CrossEncoder
        _rerankers[model] = CrossEncoder(model, max_length=512)
    return _rerankers[model]


def load_goldens(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(f"{path}:{n} missing fields {missing}")
            if row.get("review_status") != "rejected":
                rows.append(row)
    return rows


def _resolve_source(source: str) -> str:
    from rag import is_youtube_source
    if is_youtube_source(source) or os.path.isabs(source):
        return source
    return os.path.join(PROJECT_ROOT, source)


def _index_info(index, source: str) -> dict:
    import hashlib
    from rag import vector_index
    fingerprint = hashlib.sha256("\x00".join(index.chunks).encode("utf-8")).hexdigest()[:12]
    incomplete = None
    try:
        _, _, meta_path = vector_index._cache_paths(source, 200, 40, ".rag_cache")
        with open(meta_path, encoding="utf-8") as f:
            incomplete = bool(json.load(f).get("incomplete", False))
    except Exception:
        pass
    return {"chunks": len(index.chunks), "fingerprint": fingerprint, "incomplete": incomplete,
            "cache_version": vector_index._CACHE_VERSION}


def baseline_retrieve(index, question: str):
    """-> ([(chunk, score, position, is_direct_hit)] in the order search_document
    returns them, top-1 similarity score)."""
    from rag import expand_with_neighbors, retrieve_with_scores_indexed
    hits = retrieve_with_scores_indexed(index, question, k=BASELINE["k"])
    direct = {p for _, _, p in hits}
    expanded = expand_with_neighbors(index, hits, window=BASELINE["window"],
                                     min_score=BASELINE["neighbor_min_score"])
    expanded = sorted(expanded, key=lambda h: (-h[1], h[2]))  # as tools.search_document
    # Reranker filter, as tools.search_document (since 2026-10-02): keep
    # rr >= RERANK_MIN_SCORE, ordered by reranker score; unchanged if the
    # reranker is unavailable. Tuples keep their embedding score.
    from rag.reranker import rerank_scores, RERANK_MIN_SCORE
    rr = rerank_scores(question, [c for c, _, _ in expanded])
    if rr is not None:
        expanded = [expanded[i] for i in sorted((i for i, s in enumerate(rr) if s >= RERANK_MIN_SCORE),
                                                 key=lambda i: (-rr[i], i))]
    return [(c, s, p, p in direct) for c, s, p in expanded], (hits[0][1] if hits else None)


def _pages(index, position: int):
    meta = index.metadatas[position] if position < len(getattr(index, "metadatas", [])) else {}
    return meta.get("page")


def evaluate_question(row: dict, index, recall: ContextualClaimRecallMetric,
                      precision: RankedContextualPrecisionMetric) -> dict:
    from tools import search_document
    q, ideal = row["question"], row["ideal_answer"]
    retrieved, top1 = baseline_retrieve(index, q)
    sent = [c for c, _, _, _ in retrieved]

    tool_chunks = search_document.invoke({"query": q}).split("\n\n---\n\n")
    matches_tool = tool_chunks == (sent or ["No relevant content found in the document for this query."])

    case = LLMTestCase(input=q, expected_output=ideal, retrieval_context=sent)
    recall.measure(case)
    precision.measure(case)
    verdicts = precision.verdicts
    by_score = sorted(range(len(retrieved)), key=lambda i: (-retrieved[i][1], retrieved[i][2]))

    return {
        "id": row["id"], "type": row.get("type"), "difficulty": row.get("difficulty"),
        "in_document": row["in_document"], "question": q,
        "source_pages": row.get("source_pages", []),
        "recall": recall.score, "sufficient": recall.score == 1.0,
        "claims": recall.claims, "missing_claims": recall.missing_claims,
        "precision_sent_order": precision.score,
        "precision_score_order": weighted_cumulative_precision([verdicts[i]["useful"] for i in by_score]),
        "precision_at_k": precision.precision_at_k,
        "first_useful_rank_sent": precision.first_useful_rank,
        "first_useful_rank_score": next((r for r, i in enumerate(by_score, 1) if verdicts[i]["useful"]), None),
        "n_chunks": len(sent), "context_words": sum(len(c.split()) for c in sent),
        "top1_score": top1, "matches_search_document": matches_tool,
        "chunks": [
            {"rank_sent": r, "position": p, "score": round(s, 4), "direct_hit": d,
             "pages": _pages(index, p), "useful": v["useful"],
             "supported_claims": [i + 1 for i in v["supported"]], "reason": v["reason"]}
            for r, ((c, s, p, d), v) in enumerate(zip(retrieved, verdicts), 1)
        ],
    }


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(rows: List[dict]) -> dict:
    def group(rs):
        return {
            "n": len(rs),
            "recall": _mean(r["recall"] for r in rs),
            "sufficient_pct": _mean(100.0 * r["sufficient"] for r in rs),
            "precision_sent_order": _mean(r["precision_sent_order"] for r in rs),
            "precision_score_order": _mean(r["precision_score_order"] for r in rs),
            "precision_at_k": _mean(r["precision_at_k"] for r in rs),
            "first_useful_rank_sent": _mean(r["first_useful_rank_sent"] for r in rs),
            "chunks": _mean(r["n_chunks"] for r in rs),
            "context_words": _mean(r["context_words"] for r in rs),
        }
    in_doc = [r for r in rows if r["in_document"]]
    out_doc = [r for r in rows if not r["in_document"]]
    types = sorted({r["type"] for r in in_doc})
    in_top1 = sorted(r["top1_score"] for r in in_doc if r["top1_score"] is not None)
    return {
        "in_document": group(in_doc),
        "by_type": {t: group([r for r in in_doc if r["type"] == t]) for t in types},
        "out_of_document": [
            {"id": r["id"], "top1_score": r["top1_score"],
             "useful_chunks": sum(c["useful"] for c in r["chunks"]), "n_chunks": r["n_chunks"]}
            for r in out_doc
        ],
        "in_document_top1": {
            "min": in_top1[0] if in_top1 else None,
            "median": statistics.median(in_top1) if in_top1 else None,
            "max": in_top1[-1] if in_top1 else None,
        },
        "weakest": [r["id"] for r in sorted(in_doc, key=lambda r: (r["recall"], r["precision_sent_order"]))[:5]],
    }


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:.0f}%" if pct else f"{v:.2f}"


def to_markdown(meta: dict, summary: dict, rows: List[dict]) -> str:
    cols = ["n", "recall", "sufficient_pct", "precision_sent_order", "precision_score_order",
            "precision_at_k", "first_useful_rank_sent", "chunks", "context_words"]
    head = ("| group | n | recall | sufficient | precision (sent order) | precision (score order) "
            "| P@k | 1st useful rank | chunks | words |")
    lines = [f"# Retriever baseline — {meta['golden']}", "",
             f"- run: {meta['timestamp']} · judge `{meta['judge']}` · retriever {meta['retriever']}",
             f"- index: {meta['index']}", f"- search_document match: {meta['search_document_matches']}", "",
             head, "|" + "---|" * 10]

    def row_line(name, g):
        vals = [str(g["n"])] + [_fmt(g[c], pct=(c == "sufficient_pct")) for c in cols[1:]]
        return f"| {name} | " + " | ".join(vals) + " |"

    lines.append(row_line("**in-document**", summary["in_document"]))
    for t, g in summary["by_type"].items():
        lines.append(row_line(t, g))
    lines += ["", "## Out-of-document questions", "",
              f"In-document top-1 similarity: min {_fmt(summary['in_document_top1']['min'])}, "
              f"median {_fmt(summary['in_document_top1']['median'])}, "
              f"max {_fmt(summary['in_document_top1']['max'])}", ""]
    for o in summary["out_of_document"]:
        lines.append(f"- {o['id']}: top-1 similarity {_fmt(o['top1_score'])}, "
                     f"useful chunks {o['useful_chunks']}/{o['n_chunks']}")
    lines += ["", "## Weakest in-document questions", ""]
    by_id = {r["id"]: r for r in rows}
    for qid in summary["weakest"]:
        r = by_id[qid]
        got = sorted({p for c in r["chunks"] for p in (c["pages"] if isinstance(c["pages"], list) else [c["pages"]]) if p is not None})
        lines.append(f"- **{qid}** ({r['type']}) recall {_fmt(r['recall'])}, precision "
                     f"{_fmt(r['precision_sent_order'])} — gold pages {r['source_pages']}, retrieved pages {got}")
        for m in r["missing_claims"]:
            lines.append(f"  - missing: {m}")
    return "\n".join(lines) + "\n"


def sweep_retrieve(index, question: str, cfg: dict):
    """-> (chunks in document order, as search_document would send them;
    the same chunks best-first)."""
    from rag import expand_with_neighbors, retrieve_with_scores_indexed
    hits = [h for h in retrieve_with_scores_indexed(index, question, k=cfg["k"])
            if h[1] >= cfg["direct_min"]]
    if cfg["window"]:
        hits = expand_with_neighbors(index, hits, window=cfg["window"], min_score=cfg["nbr_min"])
    rr = cfg.get("rerank")
    if rr and hits:
        scores = _get_reranker(rr["model"]).predict([(question, c) for c, _, _ in hits])
        hits = sorted(((c, float(s), p) for (c, _, p), s in zip(hits, scores) if s >= rr["min_score"]),
                      key=lambda h: (-h[1], h[2]))[: rr["max_chunks"] or None]
    by_doc = [c for c, _, _ in sorted(hits, key=lambda h: h[2])]
    by_score = [c for c, _, _ in sorted(hits, key=lambda h: (-h[1], h[2]))]
    return by_doc, by_score


def sweep_question(row: dict, index, recall: ContextualClaimRecallMetric,
                   precision: RankedContextualPrecisionMetric, settings: List[dict]) -> dict:
    def case(ctx):
        return LLMTestCase(input=row["question"], expected_output=row["ideal_answer"],
                           retrieval_context=ctx)
    out = {}
    for cfg in settings:
        by_doc, by_score = sweep_retrieve(index, row["question"], cfg)
        recall.measure(case(by_doc))
        precision.measure(case(by_score))
        score_order, p_at_k = precision.score, precision.precision_at_k
        precision.measure(case(by_doc))
        out[cfg["name"]] = {
            "recall": recall.score, "sufficient": recall.score == 1.0, "precision_at_k": p_at_k,
            "precision_score_order": score_order, "precision_doc_order": precision.score,
            "useful_chunks": sum(v["useful"] for v in precision.verdicts),
            "n_chunks": len(by_doc), "context_words": sum(len(c.split()) for c in by_doc),
        }
    return out


def summarize_sweep(results: Dict[str, dict], goldens: List[dict], settings: List[dict]) -> List[dict]:
    in_ids = [g["id"] for g in goldens if g["in_document"] and g["id"] in results]
    hop_ids = [g["id"] for g in goldens if g.get("type") == "multi-hop" and g["id"] in results]
    out_ids = [g["id"] for g in goldens if not g["in_document"] and g["id"] in results]
    table = []
    for cfg in settings:
        vals = [results[i][cfg["name"]] for i in in_ids]
        recall, p_at_k = _mean(v["recall"] for v in vals), _mean(v["precision_at_k"] for v in vals)
        outs = [results[i][cfg["name"]] for i in out_ids]
        table.append({
            "config": cfg["name"], "group": cfg["group"], "recall": recall,
            "sufficient_pct": _mean(100.0 * v["sufficient"] for v in vals),
            "multi_hop_recall": _mean(results[i][cfg["name"]]["recall"] for i in hop_ids),
            "precision_at_k": p_at_k,
            "f1": round(2 * recall * p_at_k / (recall + p_at_k), 3) if recall and p_at_k else 0.0,
            "precision_score_order": _mean(v["precision_score_order"] for v in vals),
            "precision_doc_order": _mean(v["precision_doc_order"] for v in vals),
            "chunks": _mean(v["n_chunks"] for v in vals),
            "context_words": _mean(v["context_words"] for v in vals),
            "out_of_doc_useful": f"{sum(o['useful_chunks'] for o in outs)}/{sum(o['n_chunks'] for o in outs)}",
        })
    return table


def sweep_markdown(meta: dict, table: List[dict]) -> str:
    lines = [f"# Retriever recall vs precision sweep — {meta['golden']}", "",
             f"- run: {meta['timestamp']} · judge `{meta['judge']}` · index: {meta['index']}",
             f"- means over {meta['n_in_document']} in-document questions; "
             "recall = claims found, P@k = share of chunks useful, F1 of the two;",
             "  ranking precision = DeepEval weighted cumulative precision "
             "(score order = best first = what search_document sends since 2026-09-29; "
             "doc order = page order, sent before)", "",
             "| setting | recall | sufficient | multi-hop recall | P@k | F1 | ranking prec. (score order) "
             "| ranking prec. (doc order) | chunks | words | out-of-doc useful |",
             "|" + "---|" * 11]
    group = None
    for t in table:
        if t["group"] != group:
            group = t["group"]
            lines.append(f"| ***{group}*** |" + " |" * 10)
        lines.append(
            f"| {t['config']} | {_fmt(t['recall'])} | {_fmt(t['sufficient_pct'], True)} "
            f"| {_fmt(t['multi_hop_recall'])} | {_fmt(t['precision_at_k'])} | {_fmt(t['f1'])} "
            f"| {_fmt(t['precision_score_order'])} | {_fmt(t['precision_doc_order'])} "
            f"| {_fmt(t['chunks'])} | {t['context_words']:.0f} | {t['out_of_doc_useful']} |")
    return "\n".join(lines) + "\n"


def run_sweep(args, goldens: List[dict], judge: RetrievalJudge,
              recall: ContextualClaimRecallMetric, precision: RankedContextualPrecisionMetric) -> int:
    from rag import build_index
    settings = [c for c in SWEEP if not (args.core and c.get("wide"))]
    results, errors, index_meta, consecutive_errors = {}, [], {}, 0
    for source in dict.fromkeys(r["source"] for r in goldens):
        path = _resolve_source(source)
        index = build_index(path)
        index_meta[source] = _index_info(index, path)
        for row in (r for r in goldens if r["source"] == source):
            t0, calls0 = time.time(), judge.stats["new_calls"]
            try:
                results[row["id"]] = sweep_question(row, index, recall, precision, settings)
                consecutive_errors = 0
            except Exception as e:
                errors.append({"id": row["id"], "error": str(e)})
                print(f"  {row['id']}: ERROR {e}", flush=True)
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    print("  3 errors in a row (daily quota or network, see above) — stopping; "
                          "rerun later resumes from cache.")
                    break
                continue
            print(f"  {row['id']:12s} {judge.stats['new_calls'] - calls0:3d} new judge calls"
                  f"  {time.time() - t0:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    table = summarize_sweep(results, goldens, settings)
    meta = {"golden": args.golden, "timestamp": ts, "judge": judge.model.get_model_name(),
            "index": index_meta, "sweep": settings, "core_only": bool(args.core), "errors": errors,
            "n_in_document": sum(1 for g in goldens if g["in_document"] and g["id"] in results),
            "judge_usage": dict(judge.model.usage, **judge.stats)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, f"retrieval_sweep_{ts}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "table": table, "per_question": results}, f, ensure_ascii=False, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(sweep_markdown(meta, table))
    print("\n" + sweep_markdown(meta, table))
    print(f"errors: {len(errors)} · judge: {meta['judge_usage']}\nSaved {base}.json and .md")
    return 1 if errors else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Retriever recall/precision vs a golden dataset.")
    parser.add_argument("golden", help="golden JSONL, e.g. golden/dbms_u5.jsonl")
    parser.add_argument("--limit", type=int, help="only the first N questions (dry run)")
    parser.add_argument("--model", default=JUDGE_MODEL, help="Groq judge model")
    parser.add_argument("--sweep", action="store_true", help="recall vs precision tradeoff sweep")
    parser.add_argument("--core", action="store_true",
                        help="with --sweep: skip the 'wide' settings (top-6/8/10, window 2, k4 w1)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    from rag import build_index
    from tools import set_active_index

    goldens = load_goldens(args.golden)[: args.limit]
    judge = RetrievalJudge(args.model)
    recall = ContextualClaimRecallMetric(judge=judge, threshold=1.0)
    precision = RankedContextualPrecisionMetric(judge=judge, threshold=0.5)
    if args.sweep:
        return run_sweep(args, goldens, judge, recall, precision)

    rows, errors, index_meta, consecutive_errors = [], [], {}, 0
    for source in dict.fromkeys(r["source"] for r in goldens):
        path = _resolve_source(source)
        index = build_index(path)
        set_active_index(index)
        index_meta[source] = _index_info(index, path)
        for row in (r for r in goldens if r["source"] == source):
            t0 = time.time()
            try:
                res = evaluate_question(row, index, recall, precision)
                consecutive_errors = 0
            except Exception as e:
                errors.append({"id": row["id"], "error": str(e)})
                print(f"  {row['id']}: ERROR {e}", flush=True)
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    print("  3 errors in a row (daily quota or network, see above) — stopping; "
                          "rerun later resumes from cache.")
                    break
                continue
            rows.append(res)
            print(f"  {res['id']:12s} recall {res['recall']:.2f}  precision {res['precision_sent_order']:.2f}"
                  f" (score order {res['precision_score_order']:.2f})  chunks {res['n_chunks']}"
                  f"  {time.time() - t0:.0f}s")

    ts = time.strftime("%Y%m%d-%H%M%S")
    meta = {
        "golden": args.golden, "timestamp": ts, "judge": judge.model.get_model_name(),
        "retriever": f"search_document baseline {BASELINE}", "index": index_meta,
        "search_document_matches": f"{sum(r['matches_search_document'] for r in rows)}/{len(rows)}",
        "judge_usage": dict(judge.model.usage, **judge.stats), "errors": errors,
    }
    summary = summarize(rows) if rows else {}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.join(RESULTS_DIR, f"retrieval_baseline_{ts}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    if rows:
        with open(base + ".md", "w", encoding="utf-8") as f:
            f.write(to_markdown(meta, summary, rows))
        g = summary["in_document"]
        print(f"\nIn-document ({g['n']}): recall {_fmt(g['recall'])}, sufficient {_fmt(g['sufficient_pct'], True)}, "
              f"precision {_fmt(g['precision_sent_order'])} (score order {_fmt(g['precision_score_order'])}), "
              f"P@k {_fmt(g['precision_at_k'])}")
    print(f"search_document match: {meta['search_document_matches']} · errors: {len(errors)} · "
          f"judge: {meta['judge_usage']}")
    print(f"Saved {base}.json" + (" and .md" if rows else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
