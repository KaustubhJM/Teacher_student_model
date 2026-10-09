"""
evaluation/lecture_metrics.py — metrics for the LECTURE: the planner's
per-section chunks (planner.assign_context) and the teacher's 3 beats per
section (explainer.iter_section_beats). Measure only — nothing here
changes how a lecture is planned or taught. Driven by lecture_eval.py.

Section-level metrics are custom DeepEval metrics (BaseMetric, same
interface as retrieval_metrics.py) on an LLMTestCase with
    input             = the section title
    actual_output     = what the teacher says in that section (3 beats)
    retrieval_context = the chunks the planner gave that section

SectionContextPrecisionMetric — is the planner's material on-topic?
  One batched judge call per section: which chunks are relevant material
  for teaching this section. score = relevant / all chunks. Pass
  `neighbour_flags` to also get the split direct hits vs neighbours
  (neighbours are added on purpose for teaching flow).

LectureFaithfulnessMetric — does the lecture contradict the notes?
  The section text is split into the sentences the teacher speaks
  (teacher_voice.split_into_sentences). One judge call per beat labels each one
  notes / own / contradicts (the notes) / wrong (a clear error on something
  the notes don't cover) / none (filler). The beat prompt TELLS the teacher
  to add its own knowledge, so "own" is fine:
    score (faithfulness) = 1 - contradicts / factual sentences
    accuracy             = 1 - (contradicts + wrong) / factual sentences
                           ("wrong" rests on the judge's own knowledge — less
                           reliable than the notes check, so kept separate)
    from_notes share     = notes / factual sentences (informational)

teaching_depth_metric() — DeepEval GEval with fixed evaluation steps
  (mechanism/why, a concrete worked example, nuances, accuracy). 0-1.

LectureCoverage — reference-based, lecture-level: which golden ideal-answer
  claims does the lecture STATE, and in which section. One call per
  (section, batch of golden questions); claims are the retrieval eval's
  cached claims, so they're the same claims as retrieval recall.

Every judge call is cached in evaluation/.judge_cache/lecture.json, keyed
by judge model + prompt version + hashes of the inputs.
"""

import os
from typing import Dict, List, Optional, Sequence

from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

from .judge import GroqJudge
from .retrieval_metrics import JudgementCache, _sha

# Bump when a prompt below changes, so cached judgements of the old prompt
# are not reused.
PROMPT_VERSION = "v1"
LABELS_VERSION = "v2"   # sentence-label prompt only: v2 checks every clause + "wrong" label
LECTURE_JUDGE_MODEL = "qwen/qwen3.8-27b"   # not the teacher's family (gpt-oss)
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".judge_cache",
                                  "lecture.json")
CACHE_KINDS = ("precision", "faithfulness", "coverage", "depth")


class _RelevantChunks(BaseModel):
    relevant: List[int] = []
    reason: str = ""


class _Contradiction(BaseModel):
    n: int
    reason: str = ""


class _SentenceLabels(BaseModel):
    notes: List[int] = []
    own: List[int] = []
    contradicts: List[_Contradiction] = []
    wrong: List[_Contradiction] = []
    none: List[int] = []


class _Stated(BaseModel):
    stated: List[int] = []
    contradicted: List[int] = []


class LectureTemplate:
    """Judge prompts (override by subclassing; bump `version`)."""
    version = PROMPT_VERSION

    @staticmethod
    def section_precision(topic: str, section: str, chunks: Sequence[str]) -> str:
        numbered = "\n\n".join(f'[{i}]\n"""\n{c}\n"""' for i, c in enumerate(chunks, 1))
        return (
            "A lecture planner picked the numbered CHUNKS below from a student's course "
            "document as source material for ONE section of a spoken lecture.\n\n"
            f"LECTURE TOPIC: {topic}\nSECTION: {section}\n\n"
            "relevant: the numbers of the chunks that are relevant material for teaching "
            "THIS section — they explain, define, list or give an example of something "
            "the section is about. A chunk that only shares keywords, or is about a "
            "different part of the course, is NOT relevant. OCR typos and broken spacing "
            "are fine.\nreason: one short sentence about the chunks you left out.\n\n"
            'Return ONLY JSON: {"relevant": [1, 3], "reason": "..."}\n\n'
            f"CHUNKS:\n{numbered}"
        )

    @staticmethod
    def sentence_labels(section: str, notes: Sequence[str], sentences: Sequence[str]) -> str:
        notes_text = "\n\n---\n\n".join(notes) if notes else "(no notes for this section)"
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
        return (
            f"A teacher spoke the numbered SENTENCES below while teaching the section "
            f"'{section}'. The NOTES are the student's course material for this section. "
            "The teacher was told to teach from the notes AND add their own knowledge — "
            "adding things is fine. Check EVERY clause of a long sentence — one wrong "
            "clause makes the whole sentence wrong. The notes may be garbled OCR or "
            "flattened tables/diagrams; read them carefully. Label every sentence with "
            "exactly one label:\n"
            '- "notes": its facts are stated in the NOTES (same meaning in other words '
            "counts; OCR typos and broken spacing in the notes are fine)\n"
            '- "own": a factual statement that is NOT in the notes, does not conflict '
            "with them, and is correct as far as you know\n"
            '- "contradicts": some part of it states something the NOTES say differently '
            "(a real conflict, not just extra detail)\n"
            '- "wrong": some part of it is clearly factually incorrect, on something the '
            "notes don't cover (a real error, not a simplification)\n"
            '- "none": no factual claim (greeting, filler, transition, a question to the '
            "student)\n"
            "For each contradicts/wrong sentence give a short reason (what is actually "
            "true / what the notes say).\n\n"
            'Return ONLY JSON: {"notes": [1, 4], "own": [2], "contradicts": '
            '[{"n": 5, "reason": "notes say ..."}], "wrong": [{"n": 6, "reason": "..."}], '
            '"none": [3]}\n\n'
            f'NOTES:\n"""\n{notes_text}\n"""\n\nSENTENCES:\n{numbered}'
        )

    @staticmethod
    def coverage(section_text: str, groups: Sequence[dict]) -> str:
        lines, n = [], 0
        for g in groups:
            lines.append(f"Question: {g['question']}")
            for c in g["claims"]:
                n += 1
                lines.append(f"  {n}. {c}")
        return (
            "A teacher spoke the LECTURE SECTION below. The numbered CLAIMS are facts from "
            "a study guide (grouped under the question they answer).\n"
            "1. stated: numbers of the claims this section STATES (same meaning in other "
            "words counts). A claim that is only hinted at, or whose topic is merely "
            "named without that fact, is NOT stated.\n"
            "2. contradicted: numbers of the claims the section states the OPPOSITE of.\n"
            "Most claims will not be in this section — that's expected.\n\n"
            'Return ONLY JSON: {"stated": [2, 7], "contradicted": []}\n\n'
            "CLAIMS:\n" + "\n".join(lines) + f'\n\nLECTURE SECTION:\n"""\n{section_text}\n"""'
        )


DEPTH_STEPS = [
    "Check whether the Actual Output explains HOW and WHY the section's concepts work "
    "(mechanism, reasons, consequences), not only what they are called or how they are defined.",
    "Check whether it walks through at least one concrete example in real detail (specific "
    "values, steps or a scenario), not just a mention that examples exist.",
    "Check whether it covers nuances: conditions, edge cases, trade-offs, comparisons or "
    "common confusions of the topic.",
    "Check whether technical terms and statements are accurate and used precisely.",
    "Penalize vague, generic or repetitive content that could be said about any topic. Do NOT "
    "penalize the spoken, conversational style, addressing the student by name, or a question "
    "to the student at the end.",
]


class LectureJudge:
    """Groq judge + disk cache shared by all lecture metrics."""

    def __init__(self, model: Optional[str] = None, cache_path: Optional[str] = DEFAULT_CACHE_PATH,
                 template: type = LectureTemplate, llm=None):
        self.model = llm or GroqJudge(model or LECTURE_JUDGE_MODEL)
        self.template = template
        self.cache = JudgementCache(cache_path)
        for kind in CACHE_KINDS:
            self.cache.data.setdefault(kind, {})
        self.stats = {"new_calls": 0, "cache_hits": 0}

    def _prefix(self) -> str:
        return f"{self.model.get_model_name()}|{self.template.version}"

    def _cached(self, kind: str, key: str, compute):
        hit = self.cache.get(kind, key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        value = compute()
        self.stats["new_calls"] += 1
        self.cache.set(kind, key, value)
        return value

    def relevant_chunks(self, topic: str, section: str, chunks: Sequence[str]) -> dict:
        """-> {"relevant": [0-based chunk indices], "reason": str}"""
        key = f"{self._prefix()}|{_sha(topic + chr(10) + section)}|{_sha(chr(0).join(chunks))}"

        def compute():
            v = self.model.generate(self.template.section_precision(topic, section, chunks),
                                    schema=_RelevantChunks)
            return {"relevant": sorted({i - 1 for i in v.relevant if 1 <= i <= len(chunks)}),
                    "reason": v.reason.strip()}
        return self._cached("precision", key, compute)

    def sentence_labels(self, section: str, notes: Sequence[str], sentences: Sequence[str]) -> dict:
        """-> {"labels": [label per sentence, "unlabeled" if the judge skipped it],
               "reasons": {"<0-based index>": reason}}"""
        key = (f"{self._prefix()}|labels-{LABELS_VERSION}|{_sha(section)}|{_sha(chr(0).join(notes))}"
               f"|{_sha(chr(0).join(sentences))}")

        def compute():
            v = self.model.generate(self.template.sentence_labels(section, notes, sentences),
                                    schema=_SentenceLabels)
            labels = ["unlabeled"] * len(sentences)
            # lowest precedence first, so a sentence listed twice keeps the
            # more serious label (contradicts > wrong > notes > own > none)
            for name, nums in (("none", v.none), ("own", v.own), ("notes", v.notes),
                               ("wrong", [c.n for c in v.wrong]),
                               ("contradicts", [c.n for c in v.contradicts])):
                for n in nums:
                    if 1 <= n <= len(sentences):
                        labels[n - 1] = name
            reasons = {str(c.n - 1): c.reason.strip() for c in v.wrong + v.contradicts
                       if 1 <= c.n <= len(sentences)}
            return {"labels": labels, "reasons": reasons}
        return self._cached("faithfulness", key, compute)

    def stated_claims(self, section_text: str, groups: Sequence[dict]) -> dict:
        """groups = [{"id", "question", "claims"}] -> {"stated": {id: [0-based claim
        indices]}, "contradicted": {id: [...]}}"""
        key = (f"{self._prefix()}|{_sha(section_text)}|"
               f"{_sha(repr([(g['question'], g['claims']) for g in groups]))}")

        def compute():
            v = self.model.generate(self.template.coverage(section_text, groups), schema=_Stated)
            flat = [(g["id"], i) for g in groups for i in range(len(g["claims"]))]

            def regroup(nums):
                out: Dict[str, List[int]] = {}
                for n in sorted(set(nums)):
                    if 1 <= n <= len(flat):
                        gid, i = flat[n - 1]
                        out.setdefault(gid, []).append(i)
                return out
            return {"stated": regroup(v.stated), "contradicted": regroup(v.contradicted)}
        return self._cached("coverage", key, compute)

    def depth(self, topic: str, section: str, section_text: str) -> dict:
        """-> {"score": 0-1, "reason": str} via DeepEval GEval."""
        case_input = (f"A spoken lecture on '{topic}' for one student. Section being taught: "
                      f"'{section}'.")
        key = f"{self._prefix()}|{_sha(repr(DEPTH_STEPS))}|{_sha(case_input)}|{_sha(section_text)}"

        def compute():
            metric = teaching_depth_metric(self.model)
            metric.measure(LLMTestCase(input=case_input, actual_output=section_text),
                           _show_indicator=False)
            return {"score": metric.score, "reason": metric.reason}
        return self._cached("depth", key, compute)


def teaching_depth_metric(model, threshold: float = 0.5) -> GEval:
    return GEval(
        name="Teaching depth",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=DEPTH_STEPS,
        model=model, threshold=threshold, async_mode=False,
    )


class _LectureMetric(BaseMetric):
    _required_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                        SingleTurnParams.RETRIEVAL_CONTEXT]

    def __init__(self, judge: LectureJudge, topic: str = "", threshold: float = 0.5,
                 include_reason: bool = True, strict_mode: bool = False, verbose_mode: bool = False):
        self.judge, self.topic = judge, topic
        self.model = judge.model
        self.evaluation_model = self.model.get_model_name()
        self.strict_mode = strict_mode
        self.threshold = 1.0 if strict_mode else threshold
        self.using_native_model = False
        self.include_reason = include_reason
        self.verbose_mode = verbose_mode
        self.async_mode = False
        self.score = self.success = self.reason = self.error = None

    def _finish(self, score: float, reason: str) -> float:
        if self.strict_mode and score < self.threshold:
            score = 0.0
        self.score = score
        self.success = score >= self.threshold
        self.reason = reason if self.include_reason else None
        if self.verbose_mode:
            print(f"[{self.__name__}] {reason}")
        return score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return bool(self.success)


class SectionContextPrecisionMetric(_LectureMetric):
    def measure(self, test_case: LLMTestCase, *args, neighbour_flags: Optional[List[bool]] = None,
                **kwargs) -> float:
        try:
            chunks = [str(c) for c in (test_case.retrieval_context or [])]
            if not chunks:
                self.relevant_flags, self.direct_precision, self.neighbour_precision = [], None, None
                self.error = None
                return self._finish(0.0, "0.00 — the planner gave this section no chunks.")
            verdict = self.judge.relevant_chunks(self.topic, test_case.input, chunks)
            flags = [i in verdict["relevant"] for i in range(len(chunks))]
            self.relevant_flags = flags
            nb = neighbour_flags or [False] * len(chunks)
            direct = [f for f, n in zip(flags, nb) if not n]
            neigh = [f for f, n in zip(flags, nb) if n]
            self.direct_precision = sum(direct) / len(direct) if direct else None
            self.neighbour_precision = sum(neigh) / len(neigh) if neigh else None
            score = sum(flags) / len(flags)
            reason = f"{score:.2f} — {sum(flags)}/{len(flags)} chunks relevant"
            if neighbour_flags is not None:
                reason += (f" (direct {sum(direct)}/{len(direct)}, neighbours "
                           f"{sum(neigh)}/{len(neigh)})")
            if verdict["reason"] and not all(flags):
                reason += f". Left out: {verdict['reason']}"
            self.error = None
            return self._finish(score, reason)
        except Exception as e:
            self.error = str(e)
            raise

    @property
    def __name__(self):
        return "Section Context Precision"


class LectureFaithfulnessMetric(_LectureMetric):
    def measure(self, test_case: LLMTestCase, *args, beats: Optional[List[str]] = None,
                **kwargs) -> float:
        """Pass `beats` to label each beat in its own judge call: on a real
        DBMS section, one call for all ~16 sentences missed a clause-level
        error that the per-beat call caught (shorter list, closer reading)."""
        from teacher_voice import split_into_sentences   # the units the teacher speaks
        try:
            notes = [str(c) for c in (test_case.retrieval_context or [])]
            self.sentences, self.labels, flagged = [], [], []
            for unit in (beats or [test_case.actual_output or ""]):
                sents = split_into_sentences(unit)
                if not sents:
                    continue
                result = self.judge.sentence_labels(test_case.input, notes, sents)
                offset = len(self.sentences)
                self.sentences += sents
                self.labels += result["labels"]
                flagged += [(offset + int(i), r) for i, r in result["reasons"].items()]
            if not self.sentences:
                raise ValueError("test_case.actual_output (the section text) is empty")
            flagged.sort()
            self.contradictions = [{"sentence": self.sentences[i], "reason": r}
                                   for i, r in flagged if self.labels[i] == "contradicts"]
            self.wrong = [{"sentence": self.sentences[i], "reason": r}
                          for i, r in flagged if self.labels[i] == "wrong"]
            counts = {k: self.labels.count(k)
                      for k in ("notes", "own", "contradicts", "wrong", "none", "unlabeled")}
            self.counts = counts
            factual = counts["notes"] + counts["own"] + counts["contradicts"] + counts["wrong"]
            self.from_notes = counts["notes"] / factual if factual else None
            # accuracy also counts errors judged from the judge's OWN knowledge
            # (less reliable than the notes check, so it's kept separate)
            self.accuracy = (1.0 - (counts["contradicts"] + counts["wrong"]) / factual
                             if factual else 1.0)
            score = 1.0 - counts["contradicts"] / factual if factual else 1.0
            reason = (f"{score:.2f} — {counts['contradicts']} of {factual} factual sentences "
                      f"contradict the notes, {counts['wrong']} wrong on things the notes don't "
                      f"cover; {counts['notes']} from notes, {counts['own']} own knowledge, "
                      f"{counts['none']} filler")
            if counts["unlabeled"]:
                reason += f", {counts['unlabeled']} not labelled by the judge"
            self.error = None
            return self._finish(score, reason)
        except Exception as e:
            self.error = str(e)
            raise

    @property
    def __name__(self):
        return "Lecture Faithfulness"


class LectureCoverage:
    """Which golden claims does the lecture state? Not a single test case, so
    it isn't a BaseMetric: measure(sections, goldens) over the whole lecture.

    sections = [(title, text)], goldens = [{"id", "question", "claims"}]
    -> per golden: covered claim indices, contradicted ones, sections that
       taught it; score = mean over goldens of covered / all claims."""

    def __init__(self, judge: LectureJudge, batch_size: int = 10):
        self.judge, self.batch_size = judge, batch_size

    def measure(self, sections: Sequence[tuple], goldens: Sequence[dict]) -> float:
        batches = [list(goldens[i:i + self.batch_size])
                   for i in range(0, len(goldens), self.batch_size)]
        per = {g["id"]: {"stated": set(), "contradicted": set(), "sections": []} for g in goldens}
        for title, text in sections:
            for batch in batches:
                res = self.judge.stated_claims(text, batch)
                for gid, idxs in res["stated"].items():
                    per[gid]["stated"].update(idxs)
                    if idxs and title not in per[gid]["sections"]:
                        per[gid]["sections"].append(title)
                for gid, idxs in res["contradicted"].items():
                    per[gid]["contradicted"].update(idxs)
        self.per_golden = {}
        for g in goldens:
            p = per[g["id"]]
            self.per_golden[g["id"]] = {
                "coverage": len(p["stated"]) / len(g["claims"]),
                "stated": sorted(p["stated"]),
                "missed": [i for i in range(len(g["claims"])) if i not in p["stated"]],
                "contradicted": sorted(p["contradicted"]),
                "sections": p["sections"],
            }
        self.score = (sum(v["coverage"] for v in self.per_golden.values()) / len(goldens)
                      if goldens else 0.0)
        return self.score
