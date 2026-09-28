"""
quiz.py — the post-lecture Quiz agent.

Generates questions ONLY after the lecture ends, using four inputs:
  1. The FULL original document text (re-loaded fresh from disk, not
     just the chunks — chunking is lossy/fragmenting, and a quiz
     benefits from the complete picture the Planner/Explainer never
     needed all at once)
  2. The lecture outline
  3. The Planner's section-specific retrieved chunks
  4. The actual lecture transcript (what was really taught, plus every
     doubt that came up — the transcript captures what the STUDENT
     specifically engaged with, which the document alone can't)

Model: Gemini Flash first (big context, fits the whole payload), with a
fallback chain so a quota error never crashes the lecture at the very end:

  1. Gemini Flash. If it returns a PER-MINUTE rate limit, wait the delay
     Google suggests and retry once. A PER-DAY quota error (the free tier
     is only ~20 requests/day/model, and vision_describe.py spends from
     the same pool during ingestion — one call per diagram) is not
     retried, since waiting 36s won't help.
  2. Groq (the same call_llm the rest of the project uses), with a much
     smaller prompt — document/transcript/context are trimmed so the
     request fits Groq's free-tier 8000 TPM ceiling.
  3. If both fail, raise QuizUnavailable. classroom.py catches it and
     exits cleanly; because the LangGraph checkpoint is still sitting at
     generate_quiz_node, re-running the same command later resumes
     straight into quiz generation — the lecture is NOT re-taught.

Document reload also no longer depends on the file extension alone: a
path with no extension (common on Windows, where Explorer hides
extensions) is sniffed by content (PDF / docx / pptx / xlsx / plain text).
"""

import os
import re
import json
import time
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv()

from rag import load_document
from rag.ingest import LOADERS, _load_pdf

QUIZ_MODEL = "gemini-3.8-flash"
USE_REAL_GEMINI = bool(os.environ.get("GOOGLE_API_KEY"))

# Gemini-sized budget (1M context, generous TPM).
MAX_DOCUMENT_WORDS_FOR_QUIZ = 15000

# Groq-fallback budget — whole prompt has to stay well under 8000 TPM.
GROQ_MAX_DOCUMENT_WORDS = 1200
GROQ_MAX_TRANSCRIPT_WORDS = 1500
GROQ_MAX_CHUNKS_PER_SECTION = 2

# Longest we'll sleep on a per-minute rate limit before retrying once.
MAX_RETRY_WAIT_SECONDS = 65


class QuizUnavailable(RuntimeError):
    """Raised when every quiz backend failed (e.g. Gemini daily quota
    exhausted AND Groq failed). The caller should exit cleanly so the
    checkpoint stays on the quiz node and a later re-run resumes there."""


_quiz_llm = None


def _get_quiz_llm():
    global _quiz_llm
    if _quiz_llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # max_retries=0: we handle retry policy ourselves below, so a
        # daily-quota error fails fast instead of being retried pointlessly.
        _quiz_llm = ChatGoogleGenerativeAI(model=QUIZ_MODEL, temperature=0.5, max_retries=0)
    return _quiz_llm


def _extract_text_content(content) -> str:
    """ChatGoogleGenerativeAI's .content isn't always a plain string like
    ChatGroq's is — it can come back as a list of content parts (each
    either a plain string or a dict with a 'text' key). Normalize
    either shape into one string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg or "RateLimit" in type(err).__name__


def _is_daily_quota(err: Exception) -> bool:
    return "PerDay" in str(err)


def _suggested_retry_seconds(err: Exception) -> Optional[float]:
    m = re.search(r"retry in ([\d.]+)s", str(err)) or re.search(r"'retryDelay': '(\d+)s'", str(err))
    return float(m.group(1)) if m else None


def call_quiz_llm(prompt: str) -> str:
    """Gemini call with a single retry on per-minute rate limits.
    Raises on daily-quota / other errors so the caller can fall back."""
    if not USE_REAL_GEMINI:
        return json.dumps([
            {
                "question": "[offline stub — set GOOGLE_API_KEY for real quiz generation]",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "explanation": "Offline stub, no real question generated.",
            }
        ])
    try:
        return _extract_text_content(_get_quiz_llm().invoke(prompt).content)
    except Exception as e:
        if not _is_rate_limit(e) or _is_daily_quota(e):
            raise
        wait = _suggested_retry_seconds(e) or 30
        if wait > MAX_RETRY_WAIT_SECONDS:
            raise
        print(f"(Gemini rate-limited — waiting {wait:.0f}s and retrying once...)")
        time.sleep(wait + 1)
        return _extract_text_content(_get_quiz_llm().invoke(prompt).content)


def call_groq_fallback(prompt: str) -> str:
    # Imported lazily so quiz.py stays importable without langchain_groq.
    from teacher import call_llm, USE_REAL_LLM
    if not USE_REAL_LLM:
        raise RuntimeError("no GROQ_API_KEY set, Groq fallback unavailable")
    return _extract_text_content(call_llm(prompt))


class QuizQuestion(TypedDict):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n\n[...truncated for length...]"


def build_quiz_prompt(
    topic: str,
    outline: List[str],
    section_context: Dict[str, List[str]],
    transcript: List[str],
    full_document_text: str,
    num_questions: int = 8,
    max_document_words: int = MAX_DOCUMENT_WORDS_FOR_QUIZ,
    max_transcript_words: Optional[int] = None,
    max_chunks_per_section: Optional[int] = None,
) -> str:
    context_block = "\n\n".join(
        f"[{section}]\n" + "\n".join(f"- {c}" for c in chunks[:max_chunks_per_section])
        for section, chunks in section_context.items() if chunks
    )
    transcript_block = "\n".join(transcript)
    if max_transcript_words:
        transcript_block = _truncate_words(transcript_block, max_transcript_words)
    document_block = (
        _truncate_words(full_document_text, max_document_words)
        if full_document_text else "(full document unavailable — rely on the retrieved material and transcript)"
    )

    return (
        f"You just finished teaching a lecture on '{topic}'. Generate a "
        f"{num_questions}-question multiple-choice quiz testing genuine "
        "understanding of what was covered — not just trivial recall, "
        "include some questions that require connecting ideas across "
        "sections.\n\n"
        f"LECTURE OUTLINE:\n{' -> '.join(outline)}\n\n"
        f"SOURCE MATERIAL RETRIEVED PER SECTION (what the Explainer was "
        f"grounded in):\n{context_block or '(none — no document was provided)'}\n\n"
        f"FULL ORIGINAL DOCUMENT (for full context beyond what was "
        f"retrieved per-section):\n{document_block}\n\n"
        f"ACTUAL LECTURE TRANSCRIPT (what was really said, including "
        f"every question the student asked and how it was answered):\n"
        f"{transcript_block}\n\n"
        "Favor questions about things that were ACTUALLY covered in the "
        "transcript, especially anything the student asked about or "
        "seemed to need extra explanation for — that's a signal of what's "
        "worth reinforcing. Use the full document for topics the lecture "
        "may have only touched briefly.\n\n"
        "Return ONLY a JSON list, each item shaped exactly like this, "
        "nothing else:\n"
        '[{"question": "...", "options": ["...", "...", "...", "..."], '
        '"correct_answer": "the exact text of the correct option", '
        '"explanation": "1-2 sentences on why"}]'
    )


def _parse_quiz_json(raw: str) -> List[QuizQuestion]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    # Try to salvage a JSON array embedded in extra text (models
    # sometimes add a preamble or ```json fences despite instructions).
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def _load_full_document(path: str) -> str:
    """load_document(), plus content-sniffing for paths with no extension
    (ingest.py dispatches purely on extension, so 'notes' instead of
    'notes.pdf' fails with 'Unsupported file type: .').

    YouTube links go through rag.youtube_ingest — the same loader
    build_index() uses for the lecture itself."""
    from rag import is_youtube_source, load_youtube
    if is_youtube_source(path):
        return load_youtube(path)

    if os.path.splitext(path)[1]:
        return load_document(path)

    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"%PDF"):
        return "\n\n".join(e["content"] for e in _load_pdf(path))
    if head.startswith(b"PK"):
        import zipfile
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        for prefix, ext in (("word/", ".docx"), ("ppt/", ".pptx"), ("xl/", ".xlsx")):
            if any(n.startswith(prefix) for n in names):
                return LOADERS[ext](path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def generate_quiz(
    topic: str,
    outline: List[str],
    section_context: Dict[str, List[str]],
    transcript: List[str],
    document_path: Optional[str] = None,
    num_questions: int = 8,
) -> List[QuizQuestion]:
    """Generates the post-lecture quiz. Tries Gemini (full payload), then
    Groq (trimmed payload). Raises QuizUnavailable if neither produced
    a usable quiz."""
    full_document_text = ""
    if document_path:
        try:
            full_document_text = _load_full_document(document_path)
        except Exception as e:
            print(f"(couldn't reload document for quiz generation, using retrieved chunks only: {e})")

    errors = []

    # 1) Gemini — full payload.
    try:
        prompt = build_quiz_prompt(
            topic, outline, section_context, transcript, full_document_text, num_questions
        )
        questions = _parse_quiz_json(call_quiz_llm(prompt))
        if questions:
            return questions
        errors.append("Gemini returned no parseable questions")
    except Exception as e:
        reason = "daily free-tier quota used up" if _is_daily_quota(e) else type(e).__name__
        print(f"(Gemini quiz generation failed: {reason} — falling back to Groq)")
        errors.append(f"Gemini: {reason}")

    # 2) Groq — trimmed payload to fit the 8000 TPM free tier.
    try:
        small_prompt = build_quiz_prompt(
            topic, outline, section_context, transcript, full_document_text,
            num_questions=min(num_questions, 6),
            max_document_words=GROQ_MAX_DOCUMENT_WORDS,
            max_transcript_words=GROQ_MAX_TRANSCRIPT_WORDS,
            max_chunks_per_section=GROQ_MAX_CHUNKS_PER_SECTION,
        )
        questions = _parse_quiz_json(call_groq_fallback(small_prompt))
        if questions:
            return questions
        errors.append("Groq returned no parseable questions")
    except Exception as e:
        errors.append(f"Groq: {type(e).__name__}: {e}")

    raise QuizUnavailable("; ".join(errors))


def print_quiz(questions: List[QuizQuestion]) -> None:
    print("\n=== Quiz ===\n")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q.get('question', '(missing question)')}")
        for j, opt in enumerate(q.get("options", []), 1):
            print(f"   {chr(64 + j)}. {opt}")
        print(f"   Answer: {q.get('correct_answer', '?')}")
        print(f"   Why: {q.get('explanation', '')}\n")