"""
evaluation/answer_relevancy.py — AnswerRelevancyMetric: how much of an
answer actually addresses the question that was asked.

Same algorithm as DeepEval's AnswerRelevancyMetric (LLM-as-judge),
implemented locally on the project's Groq key instead of adding the
deepeval dependency (which defaults to OpenAI):

  1. The judge LLM splits the answer into standalone statements.
  2. The judge labels each statement against the question:
       yes — helps answer the question
       idk — supporting / ambiguous (analogy, background, a brief
             "good question" or "let's continue") — not penalised
       no  — about something else (a reason is given)
  3. score = (yes + idk) / number of statements, success = score >= threshold
     strict_mode: threshold 1.0 and the score is 0 unless every statement
     is relevant.

It measures relevance to the QUESTION only — not correctness, and not
whether the answer is grounded in the retrieved chunks (that's a
faithfulness metric).

The reason is built locally from the "no" verdicts instead of a third LLM
call: 2 judge calls per test case, kept small for Groq's 8000 TPM free tier.

Usage:
    from evaluation import AnswerRelevancyMetric, LLMTestCase
    metric = AnswerRelevancyMetric(threshold=0.7)
    metric.measure(LLMTestCase(input="What is a deadlock?", actual_output=answer))
    print(metric.score, metric.success, metric.reason)

    # any other judge: a function prompt -> text (Gemini, a fake for tests, ...)
    AnswerRelevancyMetric(llm=my_call_fn)

CLI — a JSON list or JSONL file of {"input": ..., "actual_output": ...}:
    python -m evaluation evaluation/sample_cases.json [--threshold 0.7] [--strict]
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
VALID_VERDICTS = ("yes", "idk", "no")

# Same rate-limit policy as explainer._call_llm_with_rate_limit_retry (not
# imported: importing explainer pulls in the TTS modules).
MAX_RATE_LIMIT_RETRIES = 2
MAX_RATE_LIMIT_WAIT_SECONDS = 65


@dataclass
class LLMTestCase:
    input: str          # the question (e.g. the student's doubt)
    actual_output: str  # the answer the RAG system gave


@dataclass
class Verdict:
    statement: str
    verdict: str        # "yes" | "idk" | "no"
    reason: str = ""


# ---------------------------------------------------------------------------
# Judge LLM
# ---------------------------------------------------------------------------

def _groq_judge(model: str) -> Callable[[str], str]:
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "AnswerRelevancyMetric needs GROQ_API_KEY in .env "
            "(or pass llm=<function prompt -> text>)."
        )
    from langchain_groq import ChatGroq
    client = ChatGroq(model=model, temperature=0)
    return lambda prompt: client.invoke(prompt).content


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return ("ratelimit" in type(e).__name__.lower() or "error code: 429" in msg
            or "rate limit" in msg or "rate_limit" in msg)


def _suggested_wait_seconds(e: Exception) -> Optional[float]:
    # Groq's message: "Please try again in 7.66s" or "... in 1m2.5s"
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _call_with_retry(llm: Callable[[str], str], prompt: str) -> str:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return llm(prompt)
        except Exception as e:
            if attempt == MAX_RATE_LIMIT_RETRIES or not _is_rate_limit(e):
                raise
            wait = _suggested_wait_seconds(e) or 30
            if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise
            print(f"(judge rate-limited — waiting {wait:.0f}s before continuing...)")
            time.sleep(wait + 1)


def _parse_json(raw: str):
    """LLMs sometimes wrap JSON in ``` fences or add a preamble — take the
    first JSON object/array in the reply."""
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    openers = sorted((text.find(o), c) for o, c in (("{", "}"), ("[", "]")) if o in text)
    for start, close in openers:
        end = text.rfind(close)
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"judge did not return JSON: {str(raw)[:200]!r}")


def _as_list(parsed, key: str) -> list:
    if isinstance(parsed, dict):
        parsed = parsed.get(key)
    if not isinstance(parsed, list):
        raise ValueError(f"judge reply has no '{key}' list")
    return parsed


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _statements_prompt(answer: str) -> str:
    return (
        "Break the ANSWER below into a list of short, standalone statements. "
        "Each statement carries one idea and makes sense on its own (replace "
        "words like 'it' or 'this' with what they refer to). Keep the "
        "answer's own wording; do not add, correct or judge anything. "
        "Greetings, encouragement and transitions are statements too.\n\n"
        'Return ONLY a JSON object: {"statements": ["...", "..."]}\n\n'
        f"ANSWER:\n{answer}"
    )


def _verdicts_prompt(question: str, statements: List[str]) -> str:
    return (
        "Decide how relevant each STATEMENT is to the QUESTION.\n"
        '- "yes": it helps answer the question (a direct answer, explanation, '
        "definition, or a detail/example about what was asked).\n"
        '- "idk": ambiguous or supporting — it does not answer the question '
        "itself but plausibly supports the answer (an analogy, related "
        "background).\n"
        '- "no": it is about something else and does not help answer the '
        "question.\n"
        "Conversational filler — greetings, praising the question, addressing "
        "the person by name, 'let's continue' — is ALWAYS \"idk\", never "
        '"no".\n'
        "Judge relevance only, not whether the statement is true.\n\n"
        f"Return ONLY a JSON object with exactly {len(statements)} verdicts, "
        "one per statement, in the same order:\n"
        '{"verdicts": [{"verdict": "yes", "reason": ""}, '
        '{"verdict": "no", "reason": "one sentence why"}]}\n'
        'Give a reason only for "no"; otherwise leave it "".\n\n'
        f"QUESTION:\n{question}\n\n"
        f"STATEMENTS:\n{json.dumps(statements, ensure_ascii=False, indent=1)}"
    )


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

class AnswerRelevancyMetric:
    def __init__(self, threshold: float = 0.5, model: str = JUDGE_MODEL,
                 llm: Optional[Callable[[str], str]] = None, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.threshold = 1.0 if strict_mode else threshold
        self.model = model
        self._llm = llm
        self.score: Optional[float] = None
        self.success: Optional[bool] = None
        self.reason: Optional[str] = None
        self.statements: List[str] = []
        self.verdicts: List[Verdict] = []

    def _get_llm(self) -> Callable[[str], str]:
        if self._llm is None:
            self._llm = _groq_judge(self.model)
        return self._llm

    def _extract_statements(self, answer: str) -> List[str]:
        raw = _call_with_retry(self._get_llm(), _statements_prompt(answer))
        statements = [str(s).strip() for s in _as_list(_parse_json(raw), "statements")]
        statements = [s for s in statements if s]
        # A non-empty answer the judge couldn't split is still one statement.
        return statements or [answer.strip()]

    def _judge_statements(self, question: str, statements: List[str]) -> List[Verdict]:
        raw = _call_with_retry(self._get_llm(), _verdicts_prompt(question, statements))
        items = _as_list(_parse_json(raw), "verdicts")
        if len(items) != len(statements):
            raise ValueError(
                f"judge returned {len(items)} verdicts for {len(statements)} statements"
            )
        verdicts = []
        for statement, item in zip(statements, items):
            if isinstance(item, dict):
                label, reason = item.get("verdict", ""), item.get("reason") or ""
            else:
                label, reason = item, ""
            label = str(label).strip().lower()
            if label not in VALID_VERDICTS:
                raise ValueError(f"judge returned an invalid verdict: {label!r}")
            verdicts.append(Verdict(statement, label, str(reason).strip() if label == "no" else ""))
        return verdicts

    def _build_reason(self) -> str:
        if not self.verdicts:
            return f"{self.score:.2f} — the answer is empty."
        irrelevant = [v for v in self.verdicts if v.verdict == "no"]
        total = len(self.verdicts)
        if not irrelevant:
            return f"{self.score:.2f} — all {total} statements address the question."
        listed = "; ".join(
            f'"{v.statement}"' + (f" ({v.reason})" if v.reason else "") for v in irrelevant
        )
        return (f"{self.score:.2f} — {total - len(irrelevant)}/{total} statements "
                f"address the question. Irrelevant: {listed}")

    def measure(self, test_case: LLMTestCase) -> float:
        if not (test_case.input or "").strip():
            raise ValueError("test_case.input (the question) is empty")
        answer = (test_case.actual_output or "").strip()

        self.statements = self._extract_statements(answer) if answer else []
        self.verdicts = (self._judge_statements(test_case.input, self.statements)
                         if self.statements else [])

        if self.verdicts:
            relevant = sum(v.verdict != "no" for v in self.verdicts)
            score = relevant / len(self.verdicts)
        else:
            score = 0.0  # empty answer answers nothing
        if self.strict_mode and score < self.threshold:
            score = 0.0

        self.score = score
        self.success = score >= self.threshold
        self.reason = self._build_reason()
        return score

    def is_successful(self) -> bool:
        return bool(self.success)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cases(path: str) -> List[LLMTestCase]:
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().strip()
    rows = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    return [LLMTestCase(input=r["input"], actual_output=r["actual_output"]) for r in rows]


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Answer relevancy (LLM-as-judge) over test cases.")
    parser.add_argument("cases", help='JSON list or JSONL of {"input": ..., "actual_output": ...}')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--model", default=JUDGE_MODEL)
    args = parser.parse_args(argv)

    cases = _load_cases(args.cases)
    metric = AnswerRelevancyMetric(threshold=args.threshold, model=args.model,
                                   strict_mode=args.strict)
    scores, failed = [], 0
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] Q: {case.input}")
        try:
            metric.measure(case)
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
            continue
        scores.append(metric.score)
        failed += not metric.success
        print(f"  {'PASS' if metric.success else 'FAIL'}  {metric.reason}")

    if scores:
        print(f"\nAverage answer relevancy: {sum(scores) / len(scores):.2f} "
              f"over {len(scores)} case(s); {failed} failed/errored "
              f"(threshold {metric.threshold})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
