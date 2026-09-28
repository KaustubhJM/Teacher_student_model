"""
planner.py — the Planner agent.

Job: see the WHOLE document, propose a lecture outline that reflects
its actual structure, and assign each section its own retrieved source
material upfront — so the Explainer starts every section already
holding curated content instead of re-deriving it mid-lecture.

Model choice: openai/gpt-oss-20b, not the openai/gpt-oss-120b used for
the Explainer in teacher.py. Planning (structuring a document into
sections) is a mechanically simpler task than holding a natural
spoken conversation, so it doesn't need the heaviest model — and
gpt-oss-20b is currently ~2x faster and ~2x cheaper on Groq's
developer tier. (Checked Groq's current model catalog while building
this: llama-3.1-8b-instant and llama-3.3-70b-versatile have both moved
to Enterprise-only pricing and are no longer available on the
free/developer tier, so those weren't options.)

Outline parsing: the model is told to return a bare JSON list, but it
sometimes wraps it in a ```json ... ``` markdown fence anyway. The old
parser then fell back to "one section per line", so the fence markers
and brackets themselves became sections ("```json", "[", "]", "```") —
that's where the random JSON-in-the-lecture and the inflated section
count came from. _parse_outline() now strips fences, pulls out the JSON
array wherever it sits, cleans any line-based fallback, and enforces
MIN/MAX_SECTIONS. If the result is still unusable it re-asks the model
once with a stricter prompt.
"""

import os
import re
import json
from typing import TypedDict, List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from rag import build_index, retrieve_with_scores_indexed, expand_with_neighbors, DocumentIndex

PLANNER_MODEL = "openai/gpt-oss-20b"
USE_REAL_LLM = bool(os.environ.get("GROQ_API_KEY"))

# Outline size limits — enforced in code, not just requested in the prompt.
MIN_SECTIONS = 4
MAX_SECTIONS = 7

# Cap on how much of the document gets fed to the Planner in one prompt.
# Groq's free/on-demand tier for openai/gpt-oss-20b caps requests at 8000
# tokens per minute — a real request with a 12000-word sample hit this
# (8668 tokens, rejected). English technical text with bullet symbols
# runs ~1.5-1.8 tokens/word, so 2500 words leaves real safety margin for
# prompt instructions + the model's response, while still giving the
# Planner substantial visibility into the document.
MAX_DOCUMENT_WORDS_FOR_PLANNING = 2500

if USE_REAL_LLM:
    from langchain_groq import ChatGroq
    _planner_llm = ChatGroq(model=PLANNER_MODEL, temperature=0.3)

    def call_planner_llm(prompt: str) -> str:
        return _planner_llm.invoke(prompt).content
else:
    def call_planner_llm(prompt: str) -> str:
        if "Return ONLY a JSON list" in prompt:
            return json.dumps(["Introduction", "Core Concepts", "Applications", "Summary"])
        return "[offline stub — set GROQ_API_KEY for real planning]"


class LecturePlan(TypedDict):
    outline: List[str]
    section_context: Dict[str, List[str]]


def _document_sample_for_planning(index: DocumentIndex, max_words: int = MAX_DOCUMENT_WORDS_FOR_PLANNING) -> str:
    """Sample chunks EVENLY across the whole document, not just its
    first portion — a document too large to fit in max_words still gets
    representation from its beginning, middle, and end, rather than the
    Planner only ever seeing whatever happens to come first."""
    if index.is_empty() or max_words <= 0:
        return ""

    total_chunks = len(index.chunks)
    avg_words_per_chunk = sum(len(c.split()) for c in index.chunks) / total_chunks
    affordable_chunks = max(1, int(max_words / max(avg_words_per_chunk, 1)))

    if affordable_chunks >= total_chunks:
        selected = index.chunks
    else:
        stride = total_chunks / affordable_chunks
        selected = [index.chunks[int(i * stride)] for i in range(affordable_chunks)]

    # Trim precisely to the word budget in case rounding let it run over.
    words_used = 0
    parts = []
    for chunk in selected:
        chunk_words = len(chunk.split())
        if words_used + chunk_words > max_words:
            break
        parts.append(chunk)
        words_used += chunk_words
    return "\n\n".join(parts)


def _is_request_too_large_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in ("too large", "rate_limit", "tokens per minute", "413"))


# ---------------------------------------------------------------------------
# Outline parsing
# ---------------------------------------------------------------------------

_JUNK_LINE = re.compile(r"^[\s\[\]{}(),`\"']*$")          # "[", "]", "```", "," ...
_FENCE_LINE = re.compile(r"^\s*```[a-zA-Z]*\s*$")          # ```json / ```
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#+)\s*")   # "- ", "1. ", "## "


def _clean_title(title: str) -> str:
    t = _LIST_PREFIX.sub("", str(title)).strip()
    t = t.strip(" ,\"'`").strip()
    return t


def _parse_outline(raw: str) -> List[str]:
    """Turn the model's reply into a clean list of section titles.
    Handles: bare JSON, ```json fenced JSON, JSON with a preamble, and
    plain bullet/numbered lists. Never returns fence markers or brackets
    as titles."""
    if not raw:
        return []
    text = raw.strip()

    # 1) Strip a markdown code fence if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # 2) Try the JSON array wherever it sits in the text.
    candidates = [text]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            # e.g. {"sections": [...]}
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
        if isinstance(parsed, list):
            titles = []
            for item in parsed:
                if isinstance(item, dict):
                    item = item.get("title") or item.get("section") or next(iter(item.values()), "")
                titles.append(_clean_title(item))
            return [t for t in titles if t]

    # 3) Line-based fallback — skip fences, brackets and other junk lines.
    titles = []
    for line in text.splitlines():
        if _FENCE_LINE.match(line) or _JUNK_LINE.match(line):
            continue
        t = _clean_title(line)
        if t:
            titles.append(t)
    return titles


def _normalize_outline(titles: List[str]) -> List[str]:
    """Dedupe (case-insensitive) and cap at MAX_SECTIONS."""
    seen, out = set(), []
    for t in titles:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[:MAX_SECTIONS]


def _outline_prompt(topic: str, sample: str, strict: bool = False) -> str:
    source = (
        f"You are planning a lecture on '{topic}' based on this source "
        f"material:\n\n{sample}\n\n"
        "Propose section titles that reflect the ACTUAL structure and "
        "content of this material, simplest concept first, building up "
        "logically."
        if sample else
        f"You are planning a lecture on '{topic}'.\n"
        "Propose section titles, simplest concept first, building up logically."
    )
    rules = (
        f"\n\nUse between {MIN_SECTIONS} and {MAX_SECTIONS} sections. "
        "Return ONLY a JSON list of section title strings, nothing else."
    )
    if strict:
        rules += (
            " Do NOT wrap it in a markdown code block, no ``` fences, no "
            'explanation. Example of the exact format: ["Title one", "Title two", '
            '"Title three", "Title four"]'
        )
    return source + rules


def build_outline_from_document(topic: str, index: DocumentIndex) -> List[str]:
    """Propose section titles based on the document's actual content
    (not just its first few chunks) — falls back to topic-only planning
    if there's no document. Automatically retries with a smaller sample
    if the API rejects the request as too large, instead of crashing.
    Retries once with a stricter prompt if the reply can't be parsed into
    MIN_SECTIONS..MAX_SECTIONS clean titles."""
    max_words = MAX_DOCUMENT_WORDS_FOR_PLANNING
    attempts = 0
    strict = False
    best: Optional[List[str]] = None

    while True:
        sample = _document_sample_for_planning(index, max_words=max_words)
        prompt = _outline_prompt(topic, sample, strict=strict)

        try:
            raw = call_planner_llm(prompt)
        except Exception as e:
            attempts += 1
            if _is_request_too_large_error(e) and attempts <= 3 and max_words > 200:
                max_words = max_words // 2
                print(f"(Planner request too large — retrying with a smaller "
                      f"document sample: ~{max_words} words)")
                continue
            raise

        outline = _normalize_outline(_parse_outline(raw))
        if len(outline) >= MIN_SECTIONS:
            return outline

        if best is None or len(outline) > len(best):
            best = outline
        if strict:
            break  # already retried once with the strict prompt
        strict = True

    # Still too short after the strict retry — use what we have, or a
    # generic outline rather than teaching zero sections.
    return best if best else [
        f"Introduction to {topic}",
        "Core Concepts",
        "How It Works in Practice",
        "Key Takeaways",
    ]


def assign_context(outline: List[str], index: DocumentIndex, k: int = 5,
                    min_score: float = 0.15, expand_window: int = 1) -> Dict[str, List[str]]:
    """For each section, retrieve the chunks most relevant to it — the
    Planner doing retrieval on the Explainer's behalf, once, upfront,
    rather than the Explainer re-retrieving mid-lecture.

    Three things this guards against:

    1. Compound section titles scoring too low to match anything.
       A title like "Foundations: Definitions, Goals, and History"
       embeds as an average of three different sub-topics, which often
       doesn't closely match any single chunk (each chunk is usually
       about just one of those things) — even when the document
       clearly covers all three. min_score=0.15 (down from the vector
       backend's general-purpose default of 0.3) accounts for this;
       compound titles legitimately score lower than single-topic ones.

    2. The same "semantically central" chunks getting vacuumed up by
       EVERY section. Retrieving independently per section with no
       memory of what earlier sections already claimed means a
       generic-sounding chunk (e.g. a document's own intro/overview
       text) can end up assigned to four different sections, while
       more specific chunks that actually belong to just one section
       get starved out. This tracks claimed chunks and prefers
       unclaimed ones, only reusing a chunk when a section genuinely
       has nothing else available above min_score.

    3. A section that genuinely spans several chunks (e.g. "Data Models
       in DBMS", which the source document covers across 5 consecutive
       sub-headings) losing its continuation just because that
       continuation chunk scored slightly lower than an unrelated
       chunk elsewhere. After picking each section's top-k chunks, we
       also pull in their immediate neighbors (by position in the
       document) as likely continuations of the same material — this
       is retrieval completeness, not a blind top_k increase: it only
       adds chunks adjacent to an already-relevant hit, never
       unrelated ones.
    """
    if index.is_empty():
        return {section: [] for section in outline}

    used_chunks: set = set()
    section_context: Dict[str, List[str]] = {}

    for section in outline:
        # Cast a wider net than k so there's room to skip already-claimed
        # chunks and still end up with k fresh ones where possible.
        candidates = retrieve_with_scores_indexed(index, section, k=k * 4)
        qualifying = [(c, s, p) for c, s, p in candidates if s >= min_score]

        fresh = [(c, s, p) for c, s, p in qualifying if c not in used_chunks][:k]
        chosen = list(fresh)

        if len(chosen) < k:
            # Not enough unclaimed material — reuse the best already-used
            # chunks rather than leaving the section thin. Better to teach
            # from overlapping material than from nothing.
            still_needed = k - len(chosen)
            chosen_texts = {c for c, s, p in chosen}
            reuse = [(c, s, p) for c, s, p in qualifying if c not in chosen_texts][:still_needed]
            chosen += reuse

        expanded = expand_with_neighbors(index, chosen, window=expand_window)
        chosen_texts = [c for c, s, p in expanded]

        used_chunks.update(chosen_texts)
        section_context[section] = chosen_texts

    return section_context


def plan_lecture(topic: str, document_path: str = None,
                  chunk_size: int = 200, overlap: int = 40) -> LecturePlan:
    """Full planning pass: index the document (if given), propose an
    outline from its actual content, and assign source material to
    each section.

    chunk_size/overlap default to the same values classroom.py uses
    (rag.build_index's defaults), so plan_demo.py previews exactly the
    outline and per-section context a real lecture would get — and
    reuses the same .rag_cache entry."""
    index = build_index(document_path, chunk_size=chunk_size, overlap=overlap) if document_path else DocumentIndex()
    outline = build_outline_from_document(topic, index)
    section_context = assign_context(outline, index)
    return {"outline": outline, "section_context": section_context}