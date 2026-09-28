"""
rag/vision_describe.py — vision-based description of diagrams, charts,
and embedded images found inside a source document.

Why this exists: OCR only recovers TEXT LABELS sitting on a diagram — it
throws away the diagram's actual structure (which box points to which,
how ideas relate), and it can't see anything in a pure image (a photo,
a chart with little/no text) at all, since OCR has nothing to extract.
A vision-capable LLM can instead look at the cropped region and describe
what it actually shows in plain language — something the Explainer can
teach from directly, not just a bag of extracted words.

Model: Gemini Flash (same family as quiz.py's QUIZ_MODEL) — already used
elsewhere in this project, has solid vision support.

Falls back to doing nothing (returns "") when GOOGLE_API_KEY isn't set —
ingest.py's existing OCR chain remains the fallback used in that case;
this module is only tried when a real key is available.

Two production issues surfaced once this ran against a real multi-page
document:
  1. The free tier caps this model at 20 requests/DAY, total — trivially
     exhausted by one document with several diagrams, and every call
     after that still burns a request against tomorrow's quota unless
     something stops trying.
  2. Re-processing the same document (e.g. resuming a classroom session
     on a different day) re-describes the exact same diagrams from
     scratch, spending quota on work already done.

A content-hash cache fixes (2). A quota circuit breaker — tripped the
moment a quota error is seen, and persisted to disk so it survives a
process restart — fixes (1).
"""

from dotenv import load_dotenv
load_dotenv()

import hashlib
import json
import os
import re
import time
from typing import List, Optional

VISION_MODEL = "gemini-3.8-flash"
USE_REAL_VISION = bool(os.environ.get("GOOGLE_API_KEY"))

_vision_llm = None

# ---------------------------------------------------------------------------
# Disk cache + quota circuit breaker
# ---------------------------------------------------------------------------

_CACHE_DIR = os.environ.get("VISION_CACHE_DIR", ".vision_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "descriptions.json")
_QUOTA_FILE = os.path.join(_CACHE_DIR, "quota_state.json")

_quota_exhausted_until: Optional[float] = None  # lazily seeded from disk


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_json(path: str, data: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def _get_cached_description(image_bytes: bytes) -> Optional[str]:
    return _load_json(_CACHE_FILE).get(_image_hash(image_bytes))


def _store_cached_description(image_bytes: bytes, description: str) -> None:
    cache = _load_json(_CACHE_FILE)
    cache[_image_hash(image_bytes)] = description
    _save_json(_CACHE_FILE, cache)


def _quota_is_exhausted() -> bool:
    """Checks whether a prior call — in this process OR an earlier one,
    since this reads from disk on first use — already hit a quota error,
    and if so whether the retry window has passed yet."""
    global _quota_exhausted_until
    if _quota_exhausted_until is None:
        _quota_exhausted_until = _load_json(_QUOTA_FILE).get("exhausted_until", 0)
    return time.time() < _quota_exhausted_until


def _seconds_until_daily_reset() -> float:
    """Gemini's per-day quota resets at midnight Pacific time. Uses fixed
    UTC-8 (PST): during daylight time that's 1h later than the real reset —
    deliberately erring on the side of not wasting requests."""
    pacific_now = time.time() - 8 * 3600
    return 86400 - (pacific_now % 86400) + 60


def _mark_quota_exhausted(error_message: str) -> None:
    """Order matters: a per-DAY quota error still carries a short 'Please
    retry in 36s' hint, but retrying after 36s just fails again — so it
    blocks until the daily reset. Otherwise (per-minute limit) Gemini's
    hint is used, or 60s if there is none (was 24h, which turned a brief
    per-minute limit into a whole day without vision)."""
    global _quota_exhausted_until
    match = re.search(r"retry in ([\d.]+)s", error_message)
    if "PerDay" in error_message:
        retry_after = _seconds_until_daily_reset()
    elif match:
        retry_after = float(match.group(1))
    else:
        retry_after = 60
    _quota_exhausted_until = time.time() + retry_after
    _save_json(_QUOTA_FILE, {"exhausted_until": _quota_exhausted_until})


def _is_quota_error(e: Exception) -> bool:
    msg = str(e)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower()


# Gemini "503 UNAVAILABLE — model is experiencing high demand": temporary,
# but while it lasts every further call also waits and fails (seen on a
# real run: ~42s lost on page-1 images). Block vision briefly instead.
OVERLOAD_BLOCK_SECONDS = 300


def _is_overloaded(e: Exception) -> bool:
    msg = str(e)
    return "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg.lower() or "overloaded" in msg.lower()


def _mark_overloaded() -> None:
    global _quota_exhausted_until
    _quota_exhausted_until = max(_quota_exhausted_until or 0, time.time() + OVERLOAD_BLOCK_SECONDS)
    _save_json(_QUOTA_FILE, {"exhausted_until": _quota_exhausted_until})


# Daily cap on vision calls made during INGESTION (describe + transcribe),
# so a big document can't use up the free tier's ~20 requests/day that the
# quiz (quiz.py, same model) also needs. Over budget -> the caller keeps
# its OCR/text fallback. Counted per Pacific day (the quota's reset clock).
VISION_DAILY_BUDGET = int(os.environ.get("VISION_DAILY_BUDGET", "12"))
_USAGE_FILE = os.path.join(_CACHE_DIR, "usage.json")
_budget_warned = False

# "Shortfall": vision was wanted during ingestion but temporarily
# unavailable (quota/503 block, daily budget used up, failed call, page
# missing from a reply). vector_index.build_index compares this counter
# before/after indexing and marks its cache INCOMPLETE, so the document is
# re-indexed once vision is available again. Not counted when there is no
# key at all (that's permanent, and caching is fine).
_shortfall = 0


def shortfall_count() -> int:
    return _shortfall


def _note_shortfall(n: int = 1) -> None:
    global _shortfall
    _shortfall += n


def vision_available_now() -> bool:
    """A vision call could be made right now: key set, no quota/overload
    block, daily budget left."""
    return USE_REAL_VISION and not _quota_is_exhausted() and _budget_left() > 0


def _pacific_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() - 8 * 3600))


def _budget_left() -> int:
    usage = _load_json(_USAGE_FILE)
    used = usage.get("calls", 0) if usage.get("day") == _pacific_day() else 0
    return VISION_DAILY_BUDGET - used


def _spend_call() -> bool:
    """Reserve one call from today's budget; False (and a one-time note)
    when it's used up."""
    global _budget_warned
    if _budget_left() <= 0:
        if not _budget_warned:
            _budget_warned = True
            print(f"(vision: today's ingestion budget of {VISION_DAILY_BUDGET} calls is used up — "
                  "keeping OCR/text for the rest; the quiz keeps its share of the quota)")
        return False
    day = _pacific_day()
    usage = _load_json(_USAGE_FILE)
    calls = usage.get("calls", 0) if usage.get("day") == day else 0
    _save_json(_USAGE_FILE, {"day": day, "calls": calls + 1})
    return True


def _handle_vision_error(e: Exception, what: str) -> None:
    if _is_quota_error(e):
        _mark_quota_exhausted(str(e))
        print(f"(vision quota exhausted for now — skipping vision {what} "
              f"for the rest of this run, falling back to OCR: {e})")
    elif _is_overloaded(e):
        _mark_overloaded()
        print(f"(vision model overloaded (503) — skipping vision for "
              f"{OVERLOAD_BLOCK_SECONDS // 60} min, falling back to OCR)")
    else:
        print(f"(vision {what} failed, falling back to OCR: {e})")


def _get_vision_llm():
    global _vision_llm
    if _vision_llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # max_retries=0: the circuit breaker below handles quota errors;
        # the library's own backoff retries would only burn time/requests.
        _vision_llm = ChatGoogleGenerativeAI(model=VISION_MODEL, temperature=0.2, max_retries=0)
    return _vision_llm


def _extract_text_content(content) -> str:
    """Same normalization quiz.py needs — ChatGoogleGenerativeAI's
    .content isn't always a plain string, it can come back as a list of
    content parts (each either a plain string or a dict with 'text')."""
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


_DESCRIBE_PROMPT = (
    "This image is a diagram, chart, flowchart, or figure taken from a "
    "student's course document, appearing between paragraphs of text. "
    "Describe what it shows clearly and completely enough that a teacher "
    "could explain it to a student without seeing the image: its overall "
    "structure, the elements/labels it contains, and — most importantly — "
    "how those elements relate to each other (e.g. what points to what, "
    "what's grouped together, what order/hierarchy/flow it shows). If it's "
    "a photo or screenshot rather than a diagram, just describe what it "
    "depicts and any text visible in it. Write 3-6 plain sentences, no "
    "preamble, no markdown."
)


def describe_image(image_bytes: bytes) -> str:
    """Returns a plain-language description of the image, or '' if
    description isn't possible (no key, quota exhausted for now, or the
    call failed — caller should fall back to its own text-based path in
    that case).

    Checks the content-hash cache first (free, no API call, survives
    across runs) and the quota circuit breaker second (skips the network
    call entirely once a quota error has been seen, until the reported
    retry window passes) before ever spending a real request.
    """
    if not USE_REAL_VISION:
        return ""

    cached = _get_cached_description(image_bytes)
    if cached is not None:
        return cached

    if _quota_is_exhausted() or not _spend_call():
        _note_shortfall()
        return ""

    try:
        import base64
        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(content=[
            {"type": "text", "text": _DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ])
        response = _get_vision_llm().invoke([message])
        description = _extract_text_content(response.content).strip()
        if description:
            _store_cached_description(image_bytes, description)
        return description
    except Exception as e:
        _handle_vision_error(e, "description")
        _note_shortfall()
        return ""


# ---------------------------------------------------------------------------
# Whole-page transcription (image-only pages OCR can't handle well)
# ---------------------------------------------------------------------------

# Bump when the prompt changes, so cached transcriptions are redone.
_TRANSCRIBE_VERSION = "t1"
_TRANSCRIBE_CACHE_FILE = os.path.join(_CACHE_DIR, "transcriptions.json")
PAGES_PER_REQUEST = 4

_TRANSCRIBE_PROMPT = (
    "You are given {n} page image(s) from a student's course material "
    "(slides or scanned notes), in order. For EACH page, write out its "
    "content so a teacher could teach from it without seeing the image:\n"
    "- Transcribe all text VERBATIM, keeping technical terms, symbols, "
    "formulas and identifiers exactly (fix only obvious spacing).\n"
    "- Reading order: title first; if the page has two columns or "
    "side-by-side blocks, write the LEFT block completely, then the RIGHT.\n"
    "- Tables: put each table between a line [TABLE] and a line [/TABLE], "
    "one line per row written as: RowLabel — Header1: value; Header2: "
    "value; ...\n"
    "- Code / pseudocode: keep its line breaks and indentation.\n"
    "- Handwriting: transcribe as best you can; write [?] for unreadable words.\n"
    "- Diagrams/figures: one line [Diagram: what it shows and how its parts relate].\n"
    "- Ignore watermarks, channel/brand logos, page numbers and decoration.\n"
    "Output exactly {n} section(s), each starting with its own line "
    "=== PAGE k === (k = 1..{n}). Plain text, no markdown."
)


def _image_key(img) -> str:
    return hashlib.md5(img.tobytes() + _TRANSCRIBE_VERSION.encode()).hexdigest()


def _jpeg_data_url(img) -> str:
    import base64
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _split_pages(text: str, n: int) -> List[str]:
    """'=== PAGE k ===' sections -> list of n strings ('' where missing)."""
    out = [""] * n
    parts = re.split(r"^\s*=== PAGE (\d+) ===\s*$", text, flags=re.M)
    for k, body in zip(parts[1::2], parts[2::2]):
        k = int(k)
        if 1 <= k <= n:
            out[k - 1] = body.strip()
    return out


def transcribe_pages(images: list) -> List[str]:
    """Verbatim transcription of whole page images (PIL), batched up to
    PAGES_PER_REQUEST pages per request. Returns one string per image; ''
    where vision isn't available (no key, quota/overload block, daily
    budget used up, call failed, page missing from the reply) — the caller
    keeps its OCR text for those. Cached per image across runs."""
    results = [""] * len(images)
    if not USE_REAL_VISION or not images:
        return results
    cache = _load_json(_TRANSCRIBE_CACHE_FILE)
    todo = []
    for i, img in enumerate(images):
        key = _image_key(img)
        if key in cache:
            results[i] = cache[key]
        else:
            todo.append((i, key, img))

    for start in range(0, len(todo), PAGES_PER_REQUEST):
        batch = todo[start:start + PAGES_PER_REQUEST]
        if _quota_is_exhausted() or not _spend_call():
            break
        try:
            from langchain_core.messages import HumanMessage
            content = [{"type": "text", "text": _TRANSCRIBE_PROMPT.format(n=len(batch))}]
            content += [{"type": "image_url", "image_url": _jpeg_data_url(img)} for _, _, img in batch]
            reply = _extract_text_content(_get_vision_llm().invoke([HumanMessage(content=content)]).content)
        except Exception as e:
            _handle_vision_error(e, "transcription")
            break
        for (i, key, _), text in zip(batch, _split_pages(reply, len(batch))):
            if text:
                results[i] = text
                cache[key] = text
        _save_json(_TRANSCRIBE_CACHE_FILE, cache)
    missing = sum(1 for i, _, _ in todo if not results[i])
    if missing:
        _note_shortfall(missing)
    return results