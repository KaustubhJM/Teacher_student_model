"""
evaluation/lecture_eval.py — evaluate the LECTURE: the planner's outline and
per-section chunks, and the teacher's 3 beats per section. MEASURE ONLY:
it calls the real functions and changes nothing in how a lecture is
planned or taught.

Step 1 — generate one real lecture (no audio), exactly as classroom does:
  build_index(path) -> planner.build_outline_from_document(topic, index)
  -> planner.assign_context(outline, index) -> per section
  explainer.teach_section_in_beats(state, section, chunks) (same prompts as
  BeatStream). While assign_context runs, planner.expand_with_neighbors is
  wrapped by a recorder that logs its input hits, so every chunk is labelled
  direct hit vs neighbour without touching planner.py. The lecture is saved
  after every section (a Groq daily limit mid-lecture loses nothing;
  --lecture FILE continues it).

Step 2 — score it (evaluation/lecture_metrics.py, judge qwen/qwen3.8-27b,
cached): section context precision, faithfulness to the notes, teaching
depth (GEval), coverage of the golden claims + a free miss diagnosis
(from the retrieval judge cache: was a chunk stating the missed claim in
the lecture's context or not), and free structure stats (words, minutes,
comprehension check, repeated sentences, unused chunks).

Run (vision off keeps the index fixed, like the other evals):
    GOOGLE_API_KEY= python -m evaluation.lecture_eval golden/dbms_u5.jsonl
    GOOGLE_API_KEY= python -m evaluation.lecture_eval golden/dbms_u5.jsonl --sections 1   # dry run
    GOOGLE_API_KEY= python -m evaluation.lecture_eval golden/dbms_u5.jsonl --lecture evaluation/results/lecture_dbms_u5_<ts>.json
"""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

from deepeval.test_case import LLMTestCase

from .generation_eval import STUDENT_NAME
from .judge import JUDGE_MODEL
from .lecture_metrics import (LECTURE_JUDGE_MODEL, LectureCoverage, LectureFaithfulnessMetric,
                              LectureJudge, SectionContextPrecisionMetric)
from .retrieval_eval import RESULTS_DIR, _index_info, _pages, _resolve_source, load_goldens
from .retrieval_metrics import RetrievalJudge, _sha

WORDS_PER_MINUTE = 145        # rough speaking rate of the neural voice (rate -5%)
DUPLICATE_COSINE = 0.9        # sentences this similar count as repeated
MIN_DUPLICATE_WORDS = 6       # ignore short filler ("Great, Kaustubh!")


def _topic_for(path: str) -> str:
    # same as classroom._derive_topic_from_filename for local files
    # (not imported: importing classroom starts the audio stack)
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_", " ").replace("-", " ").title()


def _pipeline_hash() -> str:
    """Hash of the code that shapes a lecture, so a later comparison knows
    whether the pipeline changed between two results."""
    import explainer
    import planner
    parts = [inspect.getsource(f) for f in (
        planner._document_sample_for_planning, planner._outline_prompt,
        planner.build_outline_from_document, planner.assign_context,
        explainer.build_beat_prompt, explainer.iter_section_beats)]
    parts += [repr(explainer.TEACHING_BEATS), planner.PLANNER_MODEL, explainer.EXPLAINER_MODEL,
              repr((planner.MIN_SECTIONS, planner.MAX_SECTIONS, planner.MAX_DOCUMENT_WORDS_FOR_PLANNING))]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _usage(cb) -> Dict[str, int]:
    return {model: int(u.get("input_tokens", 0) + u.get("output_tokens", 0))
            for model, u in cb.usage_metadata.items()}


def _page_numbers(index, positions) -> List[int]:
    """Sorted page numbers of these chunks (a chunk's "page" may be a list)."""
    out = set()
    for p in positions:
        v = _pages(index, p)
        out.update(x for x in (v if isinstance(v, (list, tuple)) else [v]) if x is not None)
    return sorted(out)


def _save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Step 1 — generate
# ---------------------------------------------------------------------------

def plan(index, topic: str) -> dict:
    """The real outline + assign_context, with direct-hit positions recorded."""
    import planner
    from langchain_core.callbacks import get_usage_metadata_callback

    direct_log: List[List[int]] = []
    original = planner.expand_with_neighbors

    def recording(idx, hits, *args, **kwargs):
        direct_log.append([p for _, _, p in hits])
        return original(idx, hits, *args, **kwargs)

    planner.expand_with_neighbors = recording
    try:
        with get_usage_metadata_callback() as cb:
            outline = planner.build_outline_from_document(topic, index)
            section_context = planner.assign_context(outline, index)
    finally:
        planner.expand_with_neighbors = original
    if len(direct_log) != len(outline):
        raise RuntimeError(f"recorder saw {len(direct_log)} expansions for {len(outline)} sections")

    sections = []
    for title, direct in zip(outline, direct_log):
        chunks = section_context[title]          # exactly what the teacher gets
        positions = [index.chunks.index(c) for c in chunks]
        if not set(direct) <= set(positions):
            raise RuntimeError(f"recorded hits not in the chunks of section {title!r}")
        sections.append({
            "title": title, "chunks": chunks, "positions": positions,
            "neighbour": [p not in direct for p in positions],
            "pages": [_pages(index, p) for p in positions],
            "beats": None, "teacher_tokens": None,
        })
    return {"outline": outline, "sections": sections, "planner_tokens": _usage(cb)}


def teach_missing_sections(lecture: dict, topic: str, limit: Optional[int], save) -> None:
    """Generate the beats of every section that has none yet (in order)."""
    import explainer
    from langchain_core.callbacks import get_usage_metadata_callback

    outline = lecture["outline"]
    for idx, sec in enumerate(lecture["sections"]):
        if limit is not None and idx >= limit:
            break
        if sec["beats"] is not None:
            continue
        state = {"topic": topic, "student_name": STUDENT_NAME, "outline": outline,
                 "idx": idx, "transcript": []}
        t0 = time.time()
        with get_usage_metadata_callback() as cb:
            beats = explainer.teach_section_in_beats(state, sec["title"], sec["chunks"])
        sec["beats"], sec["teacher_tokens"] = beats, _usage(cb)
        sec["generation_s"] = round(time.time() - t0, 1)
        save()
        print(f"  taught [{idx + 1}/{len(outline)}] {sec['title']}  "
              f"{sum(len(b.split()) for b in beats)} words  {sec['generation_s']}s  "
              f"tokens {sum(sec['teacher_tokens'].values())}", flush=True)


# ---------------------------------------------------------------------------
# Step 2 — score
# ---------------------------------------------------------------------------

def _section_text(sec: dict) -> str:
    return "\n\n".join(sec["beats"])


def structure_stats(lecture: dict, index) -> dict:
    """Free, deterministic checks (no judge)."""
    from teacher_voice import split_into_sentences
    from rag.vector_index import _get_model

    taught = [s for s in lecture["sections"] if s["beats"]]
    per_section = []
    for s in taught:
        words = [len(b.split()) for b in s["beats"]]
        tail = " ".join(split_into_sentences(s["beats"][-1])[-2:])
        per_section.append({"title": s["title"], "beat_words": words, "words": sum(words),
                            "minutes": round(sum(words) / WORDS_PER_MINUTE, 1),
                            "comprehension_check": "?" in tail})

    # repeated sentences across different beats (within or across sections)
    sents = [(si, bi, t) for si, s in enumerate(taught) for bi, b in enumerate(s["beats"])
             for t in split_into_sentences(b) if len(t.split()) >= MIN_DUPLICATE_WORDS]
    repeats = []
    if len(sents) > 1:
        emb = _get_model().encode([t for _, _, t in sents], normalize_embeddings=True)
        sims = emb @ emb.T
        for i in range(len(sents)):
            for j in range(i + 1, len(sents)):
                if sents[i][:2] != sents[j][:2] and sims[i, j] >= DUPLICATE_COSINE:
                    repeats.append({"a": sents[i][2], "b": sents[j][2],
                                    "sections": [taught[sents[i][0]]["title"], taught[sents[j][0]]["title"]],
                                    "cosine": round(float(sims[i, j]), 3)})

    used = [p for s in lecture["sections"] for p in s["positions"]]
    counts = {p: used.count(p) for p in set(used)}
    unused = [p for p in range(len(index.chunks)) if p not in counts]
    return {
        "sections": per_section,
        "total_words": sum(x["words"] for x in per_section),
        "total_minutes": round(sum(x["words"] for x in per_section) / WORDS_PER_MINUTE, 1),
        "comprehension_checks": sum(x["comprehension_check"] for x in per_section),
        "repeated_sentences": repeats,
        "chunks_in_index": len(index.chunks),
        "chunks_used": len(counts),
        "chunks_shared": sum(1 for c in counts.values() if c > 1),
        "unused_chunks": [{"position": p, "pages": _page_numbers(index, [p]),
                           "start": index.chunks[p][:120]} for p in unused],
        "unused_pages": _page_numbers(index, unused),
    }


def miss_diagnosis(lecture: dict, goldens: List[dict], coverage: dict, index,
                   rj: RetrievalJudge, relevant_pairs: Optional[set] = None) -> dict:
    """Free: for each claim the lecture missed, where was a chunk that states
    it (known from the retrieval judge cache)?
      right section had it, not taught  — in a section that the precision judge
                                          found it relevant to: the teacher skipped it
      only in an off-topic section      — the planner put it in the wrong section
      not retrieved by the planner      — in no taught section at all
      no known chunk states it          — e.g. ingestion lost it
    relevant_pairs = {(section title, chunk position)} judged relevant; without
    it the first two cases are reported together as "in context, not taught"."""
    lecture_positions = {p for s in lecture["sections"] if s["beats"] for p in s["positions"]}
    holders: Dict[int, List[str]] = {}
    for s in lecture["sections"]:
        if s["beats"]:
            for p in s["positions"]:
                holders.setdefault(p, []).append(s["title"])
    out = {}
    for g in goldens:
        claims = g["claims"]
        supporters: Dict[int, List[int]] = {i: [] for i in range(len(claims))}
        qa = _sha(g["question"] + chr(10) + g["ideal_answer"])
        for p, chunk in enumerate(index.chunks):
            v = rj.cache.get("chunks", f"{rj._prefix()}|{qa}|{_sha(chunk)}")
            for i in (v or {}).get("supported", []):
                supporters[i].append(p)
        rows = []
        for i in coverage[g["id"]]["missed"]:
            if any(p in lecture_positions for p in supporters[i]):
                if relevant_pairs is None:
                    why = "in context, not taught"
                elif any((t, p) in relevant_pairs for p in supporters[i] for t in holders.get(p, [])):
                    why = "right section had it, not taught"
                else:
                    why = "only in an off-topic section"
            elif supporters[i]:
                why = "not retrieved by the planner"
            else:
                why = "no known chunk states it"
            rows.append({"claim": claims[i], "why": why,
                         "pages": _page_numbers(index, supporters[i])})
        out[g["id"]] = rows
    return out


def score(lecture: dict, goldens: List[dict], index, judge: LectureJudge, rj: RetrievalJudge,
          topic: str) -> dict:
    precision = SectionContextPrecisionMetric(judge, topic=topic)
    faith = LectureFaithfulnessMetric(judge, topic=topic)
    coverage = LectureCoverage(judge)
    taught = [s for s in lecture["sections"] if s["beats"]]
    batches = [goldens[i:i + coverage.batch_size] for i in range(0, len(goldens), coverage.batch_size)]

    per_section, errors, consecutive = [], [], 0
    for n, sec in enumerate(taught, 1):
        text = _section_text(sec)
        case = LLMTestCase(input=sec["title"], actual_output=text, retrieval_context=sec["chunks"])
        before = dict(judge.model.usage)
        row = {"title": sec["title"], "chunks": len(sec["chunks"]),
               "neighbours": sum(sec["neighbour"])}
        steps = (
            ("precision", lambda: precision.measure(case, neighbour_flags=sec["neighbour"])),
            ("faithfulness", lambda: faith.measure(case, beats=sec["beats"])),
            ("depth", lambda: judge.depth(topic, sec["title"], text)),
            ("coverage", lambda: [judge.stated_claims(text, b) for b in batches]),
        )
        for name, run in steps:
            try:
                result = run()
                consecutive = 0
            except Exception as e:
                errors.append({"section": sec["title"], "metric": name, "error": str(e)})
                print(f"    {name}: ERROR {str(e)[:300]}", flush=True)
                consecutive += 1
                if consecutive >= 3:
                    break
                continue
            if name == "precision":
                row.update(relevant_positions=[sec["positions"][i] for i, ok
                                               in enumerate(precision.relevant_flags) if ok])
                row.update(precision=precision.score, direct_precision=precision.direct_precision,
                           neighbour_precision=precision.neighbour_precision,
                           irrelevant_chunks=[{"page": sec["pages"][i], "neighbour": sec["neighbour"][i],
                                               "start": sec["chunks"][i][:120]}
                                              for i, ok in enumerate(precision.relevant_flags) if not ok],
                           precision_reason=precision.reason)
            elif name == "faithfulness":
                row.update(faithfulness=faith.score, accuracy=faith.accuracy, from_notes=faith.from_notes,
                           sentence_counts=faith.counts, contradictions=faith.contradictions,
                           wrong=faith.wrong)
            elif name == "depth":
                row.update(depth=result["score"], depth_reason=result["reason"])
        if consecutive >= 3:
            print("  3 errors in a row (daily quota or network) — stopping; rerun resumes from cache.")
            break
        row["judge_tokens"] = sum(judge.model.usage[k] - before[k] for k in ("input_tokens", "output_tokens"))
        per_section.append(row)
        f = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"  scored [{n}/{len(taught)}] {sec['title']}  chunks {row['chunks']} "
              f"({row['neighbours']} neighbours)  precision {f(row.get('precision'))}  "
              f"faith {f(row.get('faithfulness'))} accuracy {f(row.get('accuracy'))} "
              f"(notes {f(row.get('from_notes'))})  "
              f"depth {f(row.get('depth'))}  judge tokens {row['judge_tokens']}", flush=True)

    result = {"sections": per_section, "errors": errors}
    if errors:
        return result     # coverage needs every (section, batch) call; rerun resumes from cache
    cov_score = coverage.measure([(s["title"], _section_text(s)) for s in taught], goldens)
    result["coverage"] = {"score": cov_score, "per_golden": coverage.per_golden}
    relevant_pairs = {(r["title"], p) for r in per_section for p in r.get("relevant_positions", [])}
    result["miss_diagnosis"] = miss_diagnosis(lecture, goldens, coverage.per_golden, index, rj,
                                              relevant_pairs)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(scores: dict, structure: dict, goldens: List[dict]) -> dict:
    secs = scores["sections"]
    out = {
        "sections_scored": len(secs),
        "precision": _mean(s.get("precision") for s in secs),
        "direct_precision": _mean(s.get("direct_precision") for s in secs),
        "neighbour_precision": _mean(s.get("neighbour_precision") for s in secs),
        "faithfulness": _mean(s.get("faithfulness") for s in secs),
        "accuracy": _mean(s.get("accuracy") for s in secs),
        "from_notes": _mean(s.get("from_notes") for s in secs),
        "contradicting_sentences": sum(len(s.get("contradictions", [])) for s in secs),
        "wrong_sentences": sum(len(s.get("wrong", [])) for s in secs),
        "depth": _mean(s.get("depth") for s in secs),
        "total_minutes": structure["total_minutes"],
        "comprehension_checks": f"{structure['comprehension_checks']}/{len(structure['sections'])}",
        "repeated_sentences": len(structure["repeated_sentences"]),
        "chunks_used": f"{structure['chunks_used']}/{structure['chunks_in_index']}",
    }
    if "coverage" in scores:
        per = scores["coverage"]["per_golden"]
        out["coverage"] = round(scores["coverage"]["score"], 3)
        out["goldens_fully_taught"] = sum(1 for v in per.values() if v["coverage"] == 1.0)
        out["goldens_not_taught"] = sum(1 for v in per.values() if v["coverage"] == 0.0)
        by_type: Dict[str, List[float]] = {}
        for g in goldens:
            by_type.setdefault(g.get("type") or "?", []).append(per[g["id"]]["coverage"])
        out["coverage_by_type"] = {t: round(statistics.mean(v), 3) for t, v in sorted(by_type.items())}
        whys = [r["why"] for rows in scores["miss_diagnosis"].values() for r in rows]
        out["missed_claims"] = {w: whys.count(w) for w in sorted(set(whys))}
    return out


def to_markdown(data: dict) -> str:
    meta, lec, sc, st, sm = (data["meta"], data["lecture"], data["scores"], data["structure"],
                             data["summary"])
    f = lambda v: "-" if v is None else f"{v:.2f}"
    md = [f"# Lecture eval — {meta['topic']}", "",
          f"golden `{meta['golden']}` · judge `{meta['judge']}` · pipeline `{meta['pipeline_hash']}` · "
          f"index `{meta['index']['fingerprint']}` · {meta['timestamp']}", "",
          "## Summary", "", "```", json.dumps(sm, indent=1), "```", "",
          "## Sections", "",
          "| # | Section | chunks (nbr) | precision (direct / nbr) | words | min | faith | accuracy | from notes | depth | check? |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    by_title = {s["title"]: s for s in sc["sections"]}
    for i, s in enumerate(st["sections"], 1):
        r = by_title.get(s["title"], {})
        md.append(f"| {i} | {s['title']} | {r.get('chunks', '-')} ({r.get('neighbours', '-')}) | "
                  f"{f(r.get('precision'))} ({f(r.get('direct_precision'))} / {f(r.get('neighbour_precision'))}) | "
                  f"{s['words']} | {s['minutes']} | {f(r.get('faithfulness'))} | {f(r.get('accuracy'))} | "
                  f"{f(r.get('from_notes'))} | {f(r.get('depth'))} | {'yes' if s['comprehension_check'] else 'NO'} |")
    md += ["", "## Contradictions with the notes (spot-check these)", ""]
    for s in sc["sections"]:
        for c in s.get("contradictions", []):
            md.append(f"- **{s['title']}** — \"{c['sentence']}\" — judge: {c['reason']}")
    md += ["", "## Errors on things the notes don't cover (judge's own knowledge — spot-check)", ""]
    for s in sc["sections"]:
        for c in s.get("wrong", []):
            md.append(f"- **{s['title']}** — \"{c['sentence']}\" — judge: {c['reason']}")
    md += ["", "## Off-topic chunks", ""]
    for s in sc["sections"]:
        for c in s.get("irrelevant_chunks", []):
            md.append(f"- **{s['title']}** p{c['page']} {'(neighbour)' if c['neighbour'] else '(direct)'}: "
                      f"{c['start']}…")
    if "coverage" in sc:
        md += ["", "## Coverage of the golden questions", "",
               "| id | coverage | taught in | missed claims (why) |", "|---|---|---|---|"]
        for gid, v in sc["coverage"]["per_golden"].items():
            whys = "; ".join(f"{r['claim']} ({r['why']})" for r in sc["miss_diagnosis"][gid])
            md.append(f"| {gid} | {v['coverage']:.2f} | {', '.join(v['sections']) or '—'} | {whys} |")
    md += ["", "## Depth reasons", ""]
    for s in sc["sections"]:
        if s.get("depth_reason"):
            md.append(f"- **{s['title']}** ({f(s.get('depth'))}): {s['depth_reason']}")
    if st["repeated_sentences"]:
        md += ["", "## Repeated sentences", ""]
        md += [f"- ({r['cosine']}) \"{r['a']}\" ≈ \"{r['b']}\"" for r in st["repeated_sentences"]]
    md += ["", f"Unused chunks: {len(st['unused_chunks'])} — pages {st['unused_pages']}", "",
           "## Outline", ""] + [f"{i}. {t}" for i, t in enumerate(lec["outline"], 1)]
    if sc["errors"]:
        md += ["", f"**{len(sc['errors'])} scoring errors** — rerun with --lecture to resume."]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a generated lecture (measure only).")
    parser.add_argument("golden")
    parser.add_argument("--lecture", help="saved lecture JSON to continue / re-score")
    parser.add_argument("--sections", type=int, help="only teach + score the first N sections (dry run)")
    parser.add_argument("--model", default=LECTURE_JUDGE_MODEL, help="Groq judge model")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    from rag import build_index

    goldens_all = load_goldens(args.golden)
    sources = sorted({g["source"] for g in goldens_all})
    if len(sources) != 1:
        raise SystemExit(f"one document per golden file expected, got {sources}")
    path = _resolve_source(sources[0])
    topic = _topic_for(path)
    index = build_index(path)
    index_meta = _index_info(index, path)

    if args.lecture:
        out_path = args.lecture
        with open(out_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data["meta"]["index"]["fingerprint"] != index_meta["fingerprint"]:
            raise SystemExit("index changed since this lecture was generated — chunks would not match")
        if data["meta"]["pipeline_hash"] != _pipeline_hash():
            print("NOTE: planner/explainer code changed since this lecture was generated")
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        stem = os.path.splitext(os.path.basename(args.golden))[0]
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"lecture_{stem}_{ts}.json")
        print(f"Planning '{topic}' ...", flush=True)
        data = {"meta": {"golden": args.golden, "source": sources[0], "topic": topic,
                         "timestamp": ts, "student_name": STUDENT_NAME, "index": index_meta,
                         "pipeline_hash": _pipeline_hash()},
                "lecture": plan(index, topic)}
        _save(out_path, data)
        print("  outline: " + " | ".join(data["lecture"]["outline"]), flush=True)

    try:
        teach_missing_sections(data["lecture"], topic, args.sections, lambda: _save(out_path, data))
    except Exception as e:
        _save(out_path, data)
        print(f"Teaching stopped: {str(e)[:300]}\nSaved what's done; continue with --lecture {out_path}")
        return 1

    rj = RetrievalJudge(JUDGE_MODEL)   # cached ideal-answer claims + chunk judgements (no new calls)
    goldens = [dict(g, claims=rj.claims(g["ideal_answer"])) for g in goldens_all if g["in_document"]]
    judge = LectureJudge(args.model)
    lecture = data["lecture"]
    if args.sections is not None:   # dry run: score only the first N sections
        lecture = dict(lecture, sections=[dict(s, beats=s["beats"] if i < args.sections else None)
                                          for i, s in enumerate(lecture["sections"])])
    scores = score(lecture, goldens, index, judge, rj, topic)
    structure = structure_stats(lecture, index)
    data.update(scores=scores, structure=structure, summary=summarize(scores, structure, goldens))
    data["meta"].update(judge=judge.model.get_model_name(), judge_usage=dict(judge.model.usage),
                        judge_stats=judge.stats, scored_at=time.strftime("%Y%m%d-%H%M%S"),
                        sections_limit=args.sections)
    _save(out_path, data)
    md_path = os.path.splitext(out_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(data))
    print("\nSummary:", json.dumps(data["summary"], indent=1))
    print(f"errors: {len(scores['errors'])} · Saved {out_path} + .md")
    return 1 if scores["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
