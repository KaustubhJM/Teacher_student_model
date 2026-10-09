"""
evaluation/retrieval_metrics.py — retriever metrics judged by an LLM,
built as custom DeepEval metrics (BaseMetric: measure / a_measure /
is_successful / __name__, threshold / strict_mode / include_reason).

Both take an LLMTestCase with
    input             = the question
    expected_output   = the golden ideal answer
    retrieval_context = the retrieved chunks, in the order the LLM gets them

ContextualClaimRecallMetric — did retrieval bring back what's needed?
  1. The judge splits the ideal answer into atomic claims (once per golden).
  2. For every chunk, the judge lists which claims it supports.
  3. score = claims supported by at least one chunk / all claims.
  recall == 1.0 means the context is sufficient to answer (ideal answers
  hold only essential claims — see golden/README.md).

RankedContextualPrecisionMetric — how much of the context is useful, and
is the useful part ranked first?
  The judge marks every chunk useful yes/no for answering the question.
  score = DeepEval's weighted cumulative precision
          (1/#useful) * sum over useful ranks k of (useful in top-k / k)
  and it also records precision@k (useful / all chunks) and the rank of
  the first useful chunk.

One judge call per (question, chunk) answers both metrics, and is cached
on disk (evaluation/.judge_cache/) keyed by judge model + prompt version +
hash(question + ideal answer) + hash(chunk text). Re-scoring the same
chunks under another k / window / min_score / order costs no new calls;
editing a golden's question or answer re-judges it automatically.

Known limit: a claim that is only supported by combining two chunks
counts as missed (claims are atomic, so this should be rare).

Usage:
    from evaluation.retrieval_metrics import (
        RetrievalJudge, ContextualClaimRecallMetric, RankedContextualPrecisionMetric)
    judge = RetrievalJudge()                      # Groq judge + disk cache
    recall = ContextualClaimRecallMetric(judge=judge, threshold=0.7)
    precision = RankedContextualPrecisionMetric(judge=judge, threshold=0.5)
    case = LLMTestCase(input=q, expected_output=ideal, retrieval_context=chunks)
    recall.measure(case); precision.measure(case)
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Union

from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

from .judge import GroqJudge, JUDGE_MODEL

# Bump when a prompt below changes, so cached judgements from the old
# prompt are not reused.
PROMPT_VERSION = "v1"
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".judge_cache",
                                  "retrieval_judgements.json")


class ClaimList(BaseModel):
    claims: List[str]


class ChunkVerdict(BaseModel):
    supported_claims: List[int] = []
    useful: str
    reason: str = ""


class ClaimRecallTemplate:
    """Judge prompts. Subclass and pass template=... to override (same idea
    as DeepEval's ContextualRecallTemplate); bump PROMPT_VERSION or change
    `version` so the cache doesn't serve answers to the old prompt."""
    version = PROMPT_VERSION

    @staticmethod
    def extract_claims(ideal_answer: str) -> str:
        return (
            "Split the ANSWER below into atomic factual claims. Each claim states "
            "exactly one fact and makes sense on its own (replace pronouns with "
            "what they refer to). Keep the answer's wording and technical terms; "
            "do not add, drop or merge facts.\n\n"
            'Return ONLY JSON: {"claims": ["...", "..."]}\n\n'
            f"ANSWER:\n{ideal_answer}"
        )

    @staticmethod
    def judge_chunk(question: str, claims: List[str], chunk: str) -> str:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
        return (
            "You are grading one CHUNK retrieved from a course document for a "
            "student's QUESTION. The CLAIMS are the facts of the ideal answer.\n\n"
            "1. supported_claims: the numbers of the claims whose fact is stated "
            "in this chunk (same meaning in other words counts; OCR typos and "
            "broken spacing are fine). A claim is NOT supported if the chunk only "
            "names the topic without stating that fact.\n"
            '2. useful: "yes" if the chunk contains information that helps answer '
            'the question (part of the ideal answer or a fact it depends on); "no" '
            "if it is about something else or only shares keywords.\n"
            "3. reason: one short sentence.\n\n"
            'Return ONLY JSON: {"supported_claims": [1, 3], "useful": "yes", '
            '"reason": "..."}\n\n'
            f"QUESTION:\n{question}\n\n"
            f"CLAIMS:\n{numbered}\n\n"
            f'CHUNK:\n"""\n{chunk}\n"""'
        )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class JudgementCache:
    """Small JSON file; saved after every new judgement so an interrupted run
    (rate limit, daily quota) keeps what it already paid for."""

    def __init__(self, path: Optional[str] = DEFAULT_CACHE_PATH):
        self.path = path
        self.data: Dict[str, dict] = {"claims": {}, "chunks": {}}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)

    def get(self, kind: str, key: str):
        return self.data[kind].get(key)

    def set(self, kind: str, key: str, value) -> None:
        self.data[kind][key] = value
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


class RetrievalJudge:
    """The per-question / per-chunk LLM judgements both metrics use."""

    def __init__(self, model: Union[str, DeepEvalBaseLLM, None] = None,
                 cache_path: Optional[str] = DEFAULT_CACHE_PATH,
                 template: type = ClaimRecallTemplate):
        self.model = model if isinstance(model, DeepEvalBaseLLM) else GroqJudge(model or JUDGE_MODEL)
        self.template = template
        self.cache = JudgementCache(cache_path)
        self.stats = {"new_calls": 0, "cache_hits": 0}

    def _prefix(self) -> str:
        return f"{self.model.get_model_name()}|{self.template.version}"

    def claims(self, ideal_answer: str) -> List[str]:
        key = f"{self._prefix()}|{_sha(ideal_answer)}"
        cached = self.cache.get("claims", key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        result = self.model.generate(self.template.extract_claims(ideal_answer), schema=ClaimList)
        claims = [c.strip() for c in result.claims if c and c.strip()]
        if not claims:
            raise ValueError("judge extracted no claims from the ideal answer")
        self.stats["new_calls"] += 1
        self.cache.set("claims", key, claims)
        return claims

    def judge_chunk(self, question: str, ideal_answer: str, chunk: str) -> dict:
        """-> {"supported": [0-based claim indices], "useful": bool, "reason": str}"""
        claims = self.claims(ideal_answer)
        key = f"{self._prefix()}|{_sha(question + chr(10) + ideal_answer)}|{_sha(chunk)}"
        cached = self.cache.get("chunks", key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        verdict = self.model.generate(self.template.judge_chunk(question, claims, chunk),
                                      schema=ChunkVerdict)
        useful = verdict.useful.strip().lower()
        if useful not in ("yes", "no"):
            raise ValueError(f"judge returned useful={verdict.useful!r}")
        supported = sorted({n - 1 for n in verdict.supported_claims if 1 <= n <= len(claims)})
        result = {
            "supported": supported,
            # a chunk that states part of the ideal answer is useful by definition
            "useful": useful == "yes" or bool(supported),
            "reason": verdict.reason.strip(),
        }
        self.stats["new_calls"] += 1
        self.cache.set("chunks", key, result)
        return result


def weighted_cumulative_precision(useful_flags: List[bool]) -> float:
    """DeepEval's contextual-precision formula: rewards useful chunks ranked
    higher. [1,0,1] -> (1/2)(1/1 + 2/3) = 0.833; [0,1] -> 0.5; none useful -> 0."""
    hits, total = 0, 0.0
    for k, useful in enumerate(useful_flags, start=1):
        if useful:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


class _RetrievalMetric(BaseMetric):
    _required_params = [SingleTurnParams.INPUT, SingleTurnParams.EXPECTED_OUTPUT,
                        SingleTurnParams.RETRIEVAL_CONTEXT]

    def __init__(self, threshold: float = 0.5, model: Union[str, DeepEvalBaseLLM, None] = None,
                 judge: Optional[RetrievalJudge] = None, include_reason: bool = True,
                 strict_mode: bool = False, verbose_mode: bool = False):
        self.strict_mode = strict_mode
        self.threshold = 1.0 if strict_mode else threshold
        self.judge = judge or RetrievalJudge(model)
        self.model = self.judge.model
        self.evaluation_model = self.model.get_model_name()
        self.using_native_model = False
        self.include_reason = include_reason
        self.verbose_mode = verbose_mode
        self.async_mode = False
        self.score = self.success = self.reason = self.error = None

    def _check(self, test_case: LLMTestCase) -> List[str]:
        if not (test_case.input or "").strip():
            raise ValueError("test_case.input (the question) is empty")
        if not (test_case.expected_output or "").strip():
            raise ValueError("test_case.expected_output (the ideal answer) is empty")
        return [str(c) for c in (test_case.retrieval_context or [])]

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


class ContextualClaimRecallMetric(_RetrievalMetric):
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        try:
            chunks = self._check(test_case)
            self.claims = self.judge.claims(test_case.expected_output)
            supported = set()
            for chunk in chunks:
                supported.update(self.judge.judge_chunk(test_case.input, test_case.expected_output,
                                                        chunk)["supported"])
            self.supported_claims = sorted(supported)
            self.missing_claims = [c for i, c in enumerate(self.claims) if i not in supported]
            score = len(supported) / len(self.claims)
            if not chunks:
                reason = f"0.00 — nothing retrieved; all {len(self.claims)} claims missing."
            elif not self.missing_claims:
                reason = f"1.00 — all {len(self.claims)} claims found in the context."
            else:
                reason = (f"{score:.2f} — {len(supported)}/{len(self.claims)} claims found. "
                          "Missing: " + "; ".join(f'"{c}"' for c in self.missing_claims))
            self.error = None
            return self._finish(score, reason)
        except Exception as e:
            self.error = str(e)
            raise

    @property
    def __name__(self):
        return "Contextual Claim Recall"


class RankedContextualPrecisionMetric(_RetrievalMetric):
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        try:
            chunks = self._check(test_case)
            self.verdicts = [self.judge.judge_chunk(test_case.input, test_case.expected_output, c)
                             for c in chunks]
            flags = [v["useful"] for v in self.verdicts]
            useful_ranks = [k for k, f in enumerate(flags, start=1) if f]
            self.precision_at_k = len(useful_ranks) / len(flags) if flags else 0.0
            self.first_useful_rank = useful_ranks[0] if useful_ranks else None
            score = weighted_cumulative_precision(flags)
            if not chunks:
                reason = "0.00 — nothing retrieved."
            elif not useful_ranks:
                reason = f"0.00 — none of the {len(flags)} chunks is useful for the question."
            else:
                reason = (f"{score:.2f} — useful chunks at ranks {useful_ranks} of {len(flags)} "
                          f"(precision@{len(flags)} = {self.precision_at_k:.2f}).")
            self.error = None
            return self._finish(score, reason)
        except Exception as e:
            self.error = str(e)
            raise

    @property
    def __name__(self):
        return "Ranked Contextual Precision"
