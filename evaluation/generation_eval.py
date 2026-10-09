"""
evaluation/generation_eval.py — evaluate the DOUBT ANSWERS the teacher gives.

For each golden question it calls the real `explainer.answer_doubt` (same
model, prompt and tools as in a lecture), records what happened, and scores:

  Faithfulness   DeepEval FaithfulnessMetric (Groq judge): are the answer's
                 claims supported by the context the teacher retrieved?
                   faithfulness  = DeepEval default: only contradictions count
                   groundedness  = strict: unsupported claims count too
                 (None when no context was retrieved — e.g. general knowledge)
  Relevancy      evaluation/answer_relevancy.py (tutor filler isn't penalised)
  Correctness    share of the golden ideal answer's claims the answer states,
                 plus claims it contradicts (claims come from the retrieval
                 judge cache, so they're the same claims as in retrieval_eval)
  Tool use       descriptive, not pass/fail (the doubt prompt lets the teacher
                 skip tools for general knowledge): did the answer rest on
                 document context; did an out-of-document question get
                 document context anyway (risk of answering from unrelated
                 slides); was web search used

Also recorded per question: the search queries the teacher wrote, the
retrieved context, latency, and teacher vs judge tokens (quota planning).

Teacher answers and judge scores are cached in
evaluation/.judge_cache/generation.json (answers keyed by question + doubt
prompt + model; scores by answer + context), so a rerun after a Groq limit
never regenerates or re-judges what is done.

Run (vision off keeps the index fixed, like retrieval_eval):
    GOOGLE_API_KEY= python -m evaluation.generation_eval golden/dbms_u5.jsonl --ids dbms_u5-002,dbms_u5-019
    GOOGLE_API_KEY= python -m evaluation.generation_eval golden/tc_u5.jsonl --limit 10
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import List, Optional

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from .answer_relevancy import AnswerRelevancyMetric, LLMTestCase as RelevancyCase
from .judge import JUDGE_MODEL
from .retrieval_eval import RESULTS_DIR, _index_info, _resolve_source, load_goldens
from .retrieval_metrics import JudgementCache, RetrievalJudge, _sha

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".judge_cache", "generation.json")
# Judge for the teacher's answers: a different model family from the teacher
# (gpt-oss-120b), so it doesn't grade its own answers, with its own Groq daily
# quota. The ideal-answer CLAIMS still come from the retrieval eval's
# gpt-oss cache, so correctness uses the same claims as retrieval recall.
GEN_JUDGE_MODEL = "qwen/qwen3.8-27b"
TRUTHS_LIMIT = 25   # fallback only: qwen's 1000 output-tokens-per-minute cap
STUDENT_NAME = "Kaustubh"
SECTION = "Revision of the whole unit"   # neutral: golden questions span every section
NO_RESULT_MARKERS = ("No relevant content found", "No document is loaded", "No web results found")


class _ClaimsStated(BaseModel):
    stated_claims: List[int] = []
    contradicted_claims: List[int] = []


def _correctness_prompt(question: str, claims: List[str], answer: str) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    return (
        "A teacher answered a student's QUESTION. The CLAIMS are the facts of the "
        "ideal answer.\n"
        "1. stated_claims: numbers of the claims the TEACHER'S ANSWER states (same "
        "meaning in other words counts). A claim the answer leaves out or only hints "
        "at is NOT stated.\n"
        "2. contradicted_claims: numbers of the claims the answer states the OPPOSITE "
        "of (a factual error).\n\n"
        'Return ONLY JSON: {"stated_claims": [1, 3], "contradicted_claims": []}\n\n'
        f"QUESTION:\n{question}\n\nCLAIMS:\n{numbered}\n\n"
        f'TEACHER\'S ANSWER:\n"""\n{answer}\n"""'
    )


class _Recorder:
    """Stands in for a tool inside explainer.answer_doubt (which looks tools up
    by name at call time) and logs every call + its output."""

    def __init__(self, tool, name: str, log: list):
        self.tool, self.name, self.log = tool, name, log

    def invoke(self, args):
        out = self.tool.invoke(args)
        self.log.append({"tool": self.name, "query": (args or {}).get("query"), "output": str(out)})
        return out


def _context_from(log: list) -> List[str]:
    pieces = []
    for call in log:
        for piece in call["output"].split("\n\n---\n\n"):
            if piece.strip() and not piece.startswith(NO_RESULT_MARKERS):
                pieces.append(piece)
    return pieces


def _topic_for(source: str) -> str:
    return os.path.splitext(os.path.basename(source))[0].replace("_", " ").replace("-", " ").strip()


def generate_answer(row: dict, cache: JudgementCache, topic: str) -> dict:
    """Run the real answer_doubt once (cached)."""
    import explainer
    from langchain_core.callbacks import get_usage_metadata_callback
    from tools import search_document, search_web

    state = {"topic": topic, "student_name": STUDENT_NAME, "outline": [], "idx": 0, "transcript": []}
    prompt_hash = _sha(explainer._doubt_system_prompt(state, SECTION))
    key = f"{explainer.EXPLAINER_MODEL}|{prompt_hash}|{_sha(row['question'])}"
    cached = cache.get("answers", key)
    if cached is not None:
        return dict(cached, cached=True)

    explainer._get_explainer_llm_with_tools()   # bind the real tool schemas before patching
    log: list = []
    original = explainer.search_document, explainer.search_web
    explainer.search_document = _Recorder(search_document, "search_document", log)
    explainer.search_web = _Recorder(search_web, "search_web", log)
    try:
        t0 = time.time()
        with get_usage_metadata_callback() as cb:
            answer = explainer.answer_doubt(state, SECTION, row["question"])
        latency = time.time() - t0
    finally:
        explainer.search_document, explainer.search_web = original
    if answer == explainer.DOUBT_FALLBACK.format(name=STUDENT_NAME):
        # answer_doubt failed and spoke its apology — report it as an error
        # (not cached, not scored as if it were a real answer).
        raise RuntimeError("answer_doubt failed; spoken fallback returned (see the printed error above)")
    usage = {"input_tokens": sum(u.get("input_tokens", 0) for u in cb.usage_metadata.values()),
             "output_tokens": sum(u.get("output_tokens", 0) for u in cb.usage_metadata.values())}
    result = {"answer": answer, "tool_calls": log, "latency_s": round(latency, 2), "teacher_usage": usage}
    cache.set("answers", key, result)
    return dict(result, cached=False)


def score_answer(row: dict, gen: dict, judge: RetrievalJudge, cache: JudgementCache,
                 claims_judge: Optional[RetrievalJudge] = None) -> dict:
    question, answer = row["question"], gen["answer"]
    context = _context_from(gen["tool_calls"])
    key = f"{judge._prefix()}|{_sha(question)}|{_sha(answer)}|{_sha(chr(0).join(context))}"
    cached = cache.get("scores", key)
    if cached is not None:
        return cached

    scores = {}
    if context:
        case = LLMTestCase(input=question, actual_output=answer, retrieval_context=context)
        faith = FaithfulnessMetric(model=judge.model, include_reason=False, async_mode=False)
        try:
            faith.measure(case)
        except Exception as e:
            # qwen allows <= 1000 OUTPUT tokens per request; listing every fact of
            # a long context can exceed that. Only then: ask for at most
            # TRUTHS_LIMIT facts, and mark the score so it's visible.
            if "OTPM" not in str(e) and "Request too large" not in str(e):
                raise
            faith = FaithfulnessMetric(model=judge.model, include_reason=False, async_mode=False,
                                       truths_extraction_limit=TRUTHS_LIMIT)
            faith.measure(case)
            scores["truths_capped"] = TRUTHS_LIMIT
        verdicts = [str(v.verdict).lower() for v in (faith.verdicts or [])]
        n = len(verdicts)
        scores["faithfulness"] = faith.score
        scores["groundedness"] = (sum(v == "yes" for v in verdicts) / n) if n else 1.0
        scores["answer_claims"] = n
        scores["unsupported_claims"] = [c for c, v in zip(faith.claims or [], verdicts) if v != "yes"]
        scores["contradicted_claims"] = [c for c, v in zip(faith.claims or [], verdicts) if v == "no"]
    else:
        scores["faithfulness"] = scores["groundedness"] = None   # nothing retrieved to be faithful to

    rel = AnswerRelevancyMetric(llm=lambda p: judge.model.generate(p))
    try:
        rel.measure(RelevancyCase(input=question, actual_output=answer))
        scores["relevancy"], scores["relevancy_reason"] = rel.score, rel.reason
    except ValueError as e:
        # e.g. the judge returned 20 verdicts for 19 statements — don't guess
        # which is which, and don't lose the question's other scores.
        scores["relevancy"], scores["relevancy_reason"] = None, f"not scored: {e}"

    claims = (claims_judge or judge).claims(row["ideal_answer"])
    stated = judge.model.generate(_correctness_prompt(question, claims, answer), schema=_ClaimsStated)
    ok = lambda nums: sorted({n for n in nums if 1 <= n <= len(claims)})
    scores["correctness"] = len(ok(stated.stated_claims)) / len(claims)
    scores["missed_ideal_claims"] = [c for i, c in enumerate(claims, 1) if i not in ok(stated.stated_claims)]
    scores["contradicted_ideal_claims"] = [claims[i - 1] for i in ok(stated.contradicted_claims)]

    cache.set("scores", key, scores)
    return scores


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the teacher's doubt answers.")
    parser.add_argument("golden")
    parser.add_argument("--ids", help="comma-separated golden ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=GEN_JUDGE_MODEL, help="Groq judge model for the answers")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    from rag import build_index
    from rag.reranker import _get_reranker
    from tools import set_active_index

    goldens = load_goldens(args.golden)
    if args.ids:
        wanted = set(args.ids.split(","))
        goldens = [g for g in goldens if g["id"] in wanted]
    goldens = goldens[: args.limit]

    cache = JudgementCache(CACHE_PATH)
    cache.data.setdefault("answers", {})
    cache.data.setdefault("scores", {})
    judge = RetrievalJudge(args.model)
    claims_judge = RetrievalJudge(JUDGE_MODEL)   # cached ideal-answer claims (same as retrieval eval)
    _get_reranker()   # load now so it doesn't count as answer latency

    rows, errors, index_meta, consecutive_errors = [], [], {}, 0
    for source in dict.fromkeys(g["source"] for g in goldens):
        path = _resolve_source(source)
        index = build_index(path)
        set_active_index(index)
        index_meta[source] = _index_info(index, path)
        topic = _topic_for(source)
        for row in (g for g in goldens if g["source"] == source):
            judge_before = dict(judge.model.usage)
            try:
                gen = generate_answer(row, cache, topic)
                sc = score_answer(row, gen, judge, cache, claims_judge)
                consecutive_errors = 0
            except Exception as e:
                errors.append({"id": row["id"], "error": str(e)})
                print(f"  {row['id']}: ERROR {str(e)[:300]}", flush=True)
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    print("  3 errors in a row (daily quota or network) — stopping; rerun resumes from cache.")
                    break
                continue
            tools = [c["tool"] for c in gen["tool_calls"]]
            judge_tokens = sum(judge.model.usage[k] - judge_before[k] for k in ("input_tokens", "output_tokens"))
            teacher_tokens = 0 if gen["cached"] else sum(gen["teacher_usage"].values())
            res = {
                "id": row["id"], "type": row.get("type"), "in_document": row["in_document"],
                "question": row["question"], "answer": gen["answer"],
                "tools": tools, "queries": [c["query"] for c in gen["tool_calls"]],
                "searched_document": "search_document" in tools, "used_web": "search_web" in tools,
                "got_document_context": any(c["tool"] == "search_document"
                                            and not c["output"].startswith(NO_RESULT_MARKERS)
                                            for c in gen["tool_calls"]),
                "latency_s": gen["latency_s"], "teacher_tokens": teacher_tokens, "judge_tokens": judge_tokens,
                **sc,
            }
            rows.append(res)
            f = lambda v: "-" if v is None else f"{v:.2f}"
            print(f"  {row['id']:12s} tools {tools or ['none']}  faith {f(res['faithfulness'])} "
                  f"grounded {f(res['groundedness'])}  relevancy {f(res['relevancy'])}  "
                  f"correct {f(res['correctness'])}  {res['latency_s']:.1f}s  "
                  f"tokens teacher {teacher_tokens} / judge {judge_tokens}", flush=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    in_doc = [r for r in rows if r["in_document"]]
    out_doc = [r for r in rows if not r["in_document"]]
    summary = {
        "n": len(rows),
        "in_document": {
            "faithfulness": _mean(r["faithfulness"] for r in in_doc),
            "groundedness": _mean(r["groundedness"] for r in in_doc),
            "relevancy": _mean(r["relevancy"] for r in in_doc),
            "correctness": _mean(r["correctness"] for r in in_doc),
            "got_document_context": _mean(1.0 * r["got_document_context"] for r in in_doc),
        },
        "out_of_document": {
            "correctness": _mean(r["correctness"] for r in out_doc),
            "relevancy": _mean(r["relevancy"] for r in out_doc),
            "got_document_context": _mean(1.0 * r["got_document_context"] for r in out_doc),
            "used_web": _mean(1.0 * r["used_web"] for r in out_doc),
        },
        "latency_s_mean": _mean(r["latency_s"] for r in rows),
        "teacher_tokens_per_q": _mean(r["teacher_tokens"] for r in rows if r["teacher_tokens"]),
        "judge_tokens_per_q": _mean(r["judge_tokens"] for r in rows if r["judge_tokens"]),
    }
    meta = {"golden": args.golden, "timestamp": ts, "judge": judge.model.get_model_name(),
            "index": index_meta, "section": SECTION, "errors": errors}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"generation_{ts}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=1)
    print("\nSummary:", json.dumps(summary, indent=1))
    print(f"errors: {len(errors)} · Saved {path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
