"""
rag/ingest.py — turn a source file into ordered, structured elements, then
into overlapping text chunks.

Supports: .pdf, .md, .txt, .pptx, .docx, .csv, .xlsx/.xls, .html/.htm,
.json, and images (.png/.jpg/.jpeg) via OCR.

Two entry points:
  - load_document(path) -> str                    (back-compat, flat text)
  - load_document_structured(path) -> List[dict]   (ordered, typed elements)

Structured elements look like:
  {"type": "text" | "table" | "diagram" | "diagram_placeholder" | "image",
   "content": str, "page": int | None, "top": float}

Diagrams/flowcharts (detected inline between paragraphs) and standalone
embedded images are described using a vision-capable LLM (see
vision_describe.py) when GOOGLE_API_KEY is set, so the Explainer gets an
actual description of structure/content instead of just OCR'd labels.
Falls back to the OCR-based recovery chain when no key is available.

chunk_elements() packs these into metadata-carrying chunks; chunk_text()
is a thin back-compat wrapper for plain strings.
"""

import json as _json
import os
import re
import string
from typing import List, Optional
from .vision_describe import describe_image, transcribe_pages, USE_REAL_VISION
from .page_ocr import ocr_pages, transcribe_pages_with_vision, ocr_available, ocr_image_text, ocr_image_words


def _sniff_extension(path: str) -> str:
    """For a path with no extension (common on Windows, where Explorer
    hides them): guess the type from the file's first bytes. Falls back to
    '.txt'."""
    with open(path, "rb") as f:
        head = f.read(512)
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith(b"\x89PNG"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"PK"):
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            names = []
        for prefix, ext in (("word/", ".docx"), ("ppt/", ".pptx"), ("xl/", ".xlsx")):
            if any(n.startswith(prefix) for n in names):
                return ext
    lowered = head.lower()
    if b"<html" in lowered or b"<!doctype html" in lowered:
        return ".html"
    return ".txt"


def _file_extension(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else _sniff_extension(path)


def is_pdf_file(path: str) -> bool:
    """True for a local PDF (by extension or sniffed content); False for
    other files and for URLs (e.g. YouTube)."""
    return os.path.isfile(path) and _file_extension(path) == ".pdf"


def missing_pdf_pages(path: str, metadatas: List[dict]) -> tuple:
    """(total_pages, pages_that_contributed_nothing) for a PDF, using the
    page numbers chunk_elements() stores in each chunk's metadata — so it
    works for a freshly built AND a cached index. A missing page is one
    whose text couldn't be extracted at all (typically image-only pages
    where OCR/vision were unavailable)."""
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
    covered = set()
    for m in metadatas:
        page = m.get("page")
        if isinstance(page, list):
            covered.update(page)
        elif page is not None:
            covered.add(page)
    return total, [n for n in range(1, total + 1) if n not in covered]


def load_document(path: str) -> str:
    """Extract raw text from a source file, flattened to one string."""
    ext = _file_extension(path)
    if ext == ".pdf":
        elements = _load_pdf(path)
        return "\n\n".join(e["content"] for e in elements)
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {', '.join(sorted(LOADERS) + ['.pdf'])}"
        )
    return loader(path)


def load_document_structured(path: str) -> List[dict]:
    """Like load_document, but returns an ordered list of typed elements
    instead of one flat string, so downstream chunking can keep tables and
    diagrams distinguished from prose and preserve their original position
    in the document instead of flattening everything together."""
    ext = _file_extension(path)
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in _IMAGE_EXTS:
        return _load_image_structured(path)
    # Other formats don't (yet) distinguish tables/diagrams from prose at
    # the loader level — wrap their plain text as ordered "text" elements so
    # chunk_elements / metadata / expansion work the same regardless of
    # source format.
    text = load_document(path)
    return [
        {"type": "text", "content": unit, "page": None, "top": i}
        for i, unit in enumerate(_split_into_units(text))
    ]


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def _read_text_file(path: str) -> str:
    """UTF-8 (BOM-tolerant) first, then Windows-1252 — what Notepad/older
    Windows tools save by default, which strict UTF-8 crashed on — then
    UTF-8 with undecodable bytes replaced, so a text file never crashes
    ingestion."""
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_text(path: str) -> str:
    return _read_text_file(path)


# ---------------------------------------------------------------------------
# Malformed-extraction detection
# ---------------------------------------------------------------------------
def _looks_like_diagram(words: List[dict], text: str) -> bool:
    """
    Heuristically detects whether a group of PDF words looks like a
    diagram/flowchart rather than normal paragraph text.

    Layout-based detection. The key signal is WORDS PER LINE: a diagram's
    labels are short, isolated phrases, so each line carries only a few
    words. Normal prose wraps to fill most of the page width, so its
    lines carry many more words per line, even though individual WORDS in
    prose are just as short as individual words in a label (almost every
    English word is under 20 characters). That's why word-length/short-word
    ratios alone (the old signal) fired on ordinary paragraphs too — they
    don't distinguish "a paragraph made of short words" from "a diagram
    made of short labels". Words-per-line does.

    This is intentionally generic — it does not look for DBMS-specific
    words or diagram names.
    """

    # We need several words before calling something a diagram.
    if len(words) < 4:
        return False

    # Extract the actual text from the PDF word objects.
    texts = [
        str(w.get("text", "")).strip()
        for w in words
        if str(w.get("text", "")).strip()
    ]

    if len(texts) < 4:
        return False

    # Primary signal: average words per line. Diagram labels sit a few
    # words to a line; wrapped prose paragraphs run much higher (typically
    # 8-15+ words/line). This is what the old heuristic was missing —
    # without it, ordinary multi-line prose satisfied every other check.
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    words_per_line = [len(l.split()) for l in lines]
    avg_words_per_line = sum(words_per_line) / len(words_per_line)
    if avg_words_per_line > 6:
        return False

    # Calculate the average length of each extracted word/label.
    avg_word_length = sum(len(t) for t in texts) / len(texts)

    # Get the vertical positions of the extracted words.
    # PDF coordinates use "top" for the vertical position.
    tops = [
        float(w.get("top", 0))
        for w in words
        if w.get("top") is not None
    ]

    if len(tops) < 4:
        return False

    # Calculate how much vertical space the group occupies.
    vertical_span = max(tops) - min(tops)

    # A collection of labels distributed vertically is more likely
    # to be a diagram than a normal single-line sentence.
    vertically_spread = vertical_span >= 25

    # Final heuristic: short lines (few words each), short words on
    # average, spread out vertically.
    return (
        avg_words_per_line <= 6
        and avg_word_length <= 15
        and vertically_spread
    )

# Operators found in code / pseudocode / formulas ("X := X + Y;",
# "read(X);", "A <- B"). Such text is also made of short tokens, but
# chopped diagram labels (": Cl; : ie; : n;") contain none of them.
_CODE_OR_MATH = re.compile("[=+<>\u2190\u2264\u2265\u2260]|\\w\\(")


def is_garbled_text(text: str, min_len: int = 20, min_tokens: int = 6) -> bool:
    """Heuristic check for character-by-character extraction garbage — the
    ': Cl; : ie; : n; : T; : he;' pattern seen when a diagram's text labels
    get chopped into 1-3 character fragments by PDF text extraction.

    Tuned specifically to this failure mode (many very short,
    punctuation-separated tokens), not a general gibberish detector. Guards
    against false positives on short real headings that happen to contain
    ratio notation like "(1:1):" by requiring a minimum token count —
    genuine garbled diagram text has many fragments, a heading has few.
    Code/pseudocode/formulas (at least one operator per 10 tokens) are
    never garbled — before, "X:= X+ Y; Y := X + Y;" was flagged, which
    sends a text-layer code block to the diagram path (vision summary or
    placeholder instead of the code).
    """
    if not text or len(text) < min_len:
        return False
    tokens = text.split()
    if len(tokens) < min_tokens:
        return False
    if len(_CODE_OR_MATH.findall(text)) >= 0.1 * len(tokens):
        return False
    stripped = [t.strip(string.punctuation) for t in tokens]
    stripped = [t for t in stripped if t]
    if not stripped:
        return True
    avg_len = sum(len(t) for t in stripped) / len(stripped)
    short_ratio = sum(1 for t in stripped if len(t) <= 3) / len(stripped)
    punct_ratio = sum(1 for c in text if c in ":;") / len(text)
    return (punct_ratio > 0.06 and short_ratio > 0.55) or (avg_len < 2.0 and len(stripped) >= min_tokens)


# ---------------------------------------------------------------------------
# PDF — order-preserving text/table/diagram extraction
# ---------------------------------------------------------------------------

def _group_words_into_blocks(words: List[dict], line_gap: float = 3.0,
                              block_gap: float = 8.0) -> List[dict]:
    """Group pdfplumber words into line-then-paragraph blocks using their
    y-position, so each block can be individually quality-checked and
    ordered — instead of pdfplumber's own extract_text(), which flattens
    everything (including a mis-extracted diagram's fragments) into one
    blob with no way to isolate or reorder the bad part."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines, current, current_top = [], [words[0]], words[0]["top"]
    for w in words[1:]:
        if abs(w["top"] - current_top) <= line_gap:
            current.append(w)
        else:
            lines.append(current)
            current, current_top = [w], w["top"]
    lines.append(current)

    line_infos = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w["x0"])
        line_infos.append({
            "text": " ".join(w["text"] for w in line_sorted),
            "top": min(w["top"] for w in line),
            "bottom": max(w["bottom"] for w in line),
            "x0": min(w["x0"] for w in line),
            "x1": max(w["x1"] for w in line),
    
            # Keep the original pdfplumber word objects and their coordinates.
            # The diagram detector will use this layout information later.
            "words": line_sorted,
        })
    
    blocks, current_block = [], [line_infos[0]]
    for li in line_infos[1:]:
        if li["top"] - current_block[-1]["bottom"] <= block_gap:
            current_block.append(li)
        else:
            blocks.append(current_block)
            current_block = [li]
    blocks.append(current_block)
    
    return [
        {
            "text": "\n".join(li["text"] for li in b),
            "top": min(li["top"] for li in b),
            "bottom": max(li["bottom"] for li in b),
            "x0": min(li["x0"] for li in b),
            "x1": max(li["x1"] for li in b),
    
            # Preserve all words belonging to this block.
            "words": [
                word
                for li in b
                for word in li["words"]
            ],
        }
        for b in blocks
    ]


def _word_in_bbox(word: dict, bboxes: List[tuple], pad: float = 2.0) -> bool:
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    for (x0, top, x1, bottom) in bboxes:
        if x0 - pad <= cx <= x1 + pad and top - pad <= cy <= bottom + pad:
            return True
    return False


def _find_real_tables(page) -> List[dict]:
    """pdfplumber's default line-based table detection false-positives
    badly on documents with a decorative ruled/grid background (e.g.
    notebook-paper-style notes) — it detects the background grid itself as
    one giant table spanning almost the entire page. Gate detected tables
    on plausible size/shape/density so that only genuinely tabular content
    gets pulled out as a table, and a decorative background doesn't
    swallow the whole page's text as "inside a table"."""
    try:
        tables = page.find_tables()
    except Exception:
        return []
    real = []
    for t in tables:
        x0, top, x1, bottom = t.bbox
        area_ratio = ((x1 - x0) * (bottom - top)) / max(page.width * page.height, 1)
        try:
            rows = t.extract() or []
        except Exception:
            rows = []
        if not rows or not (2 <= len(rows) <= 25) or not (2 <= len(rows[0]) <= 10):
            continue
        if area_ratio >= 0.5:
            continue
        real.append({"bbox": t.bbox, "rows": rows})
    return real


def _reconstruct_diagram_text(page, bbox: tuple) -> str:
    """Retry extraction over just this region with much looser word-joining
    tolerance. Diagram labels are often laid out as individually-positioned
    glyphs (a common export artifact from diagramming tools), which
    pdfplumber's default tight tolerance chops into 1-3 character fragments
    — the ": Cl; : ie;" symptom. A wider tolerance re-joins those fragments
    into real words far more often than the default pass does."""
    x0, top, x1, bottom = bbox
    pad = 3
    try:
        region = page.within_bbox((
            max(x0 - pad, 0), max(top - pad, 0),
            min(x1 + pad, page.width), min(bottom + pad, page.height),
        ))
        words = region.extract_words(x_tolerance=8, y_tolerance=6)
    except Exception:
        return ""
    blocks = _group_words_into_blocks(words, line_gap=4.0, block_gap=1000.0)
    return " / ".join(b["text"].replace("\n", " ") for b in blocks)


def _ocr_region(page, bbox: tuple) -> str:

    """Rasterize just this bbox and OCR it — the fallback when vision isn't
    available (figures) or even the loose-tolerance reconstruction still
    looks garbled. Uses pytesseract if it's actually usable, otherwise the
    local RapidOCR engine (page_ocr); '' if neither is available."""
    try:
        x0, top, x1, bottom = bbox
        pad = 3
        cropped = page.within_bbox((
            max(x0 - pad, 0), max(top - pad, 0),
            min(x1 + pad, page.width), min(bottom + pad, page.height),
        ))
        image = cropped.to_image(resolution=200).original
    except Exception:
        return ""
    try:
        import pytesseract
        return pytesseract.image_to_string(image).strip()
    except Exception:
        pass  # not installed, or the tesseract binary is missing
    try:
        return ocr_image_text(image)
    except Exception:
        return ""

def _render_region_bytes(page, bbox: tuple, pad: float = 3) -> "bytes | None":
    """Rasterize a bbox region of the page to PNG bytes, for feeding to a
    vision model. Returns None if rendering isn't possible (Pillow/
    pdfplumber's image backend missing, or any other failure) so callers
    fall back to the text-based path instead of crashing."""
    try:
        import io
        x0, top, x1, bottom = bbox
        cropped = page.within_bbox((
            max(x0 - pad, 0), max(top - pad, 0),
            min(x1 + pad, page.width), min(bottom + pad, page.height),
        ))
        image = cropped.to_image(resolution=200).original
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _bboxes_overlap(a: tuple, b: tuple) -> bool:
    ax0, atop, ax1, abottom = a
    bx0, btop, bx1, bbottom = b
    return ax0 < bx1 and ax1 > bx0 and atop < bbottom and abottom > btop


def _find_column_gutter(words: List[dict], allow_unaligned: bool = False) -> Optional[tuple]:
    """(gutter_x0, gutter_x1) if the page's text forms two side-by-side
    prose columns, else None — and None means the page is processed exactly
    as before (row-by-row), so single-column pages are unaffected.

    Without this, pdfplumber rows spanning the page glued each left-column
    line to the right-column line at the same height ("L1 ... R1 ...").

    Gutter = the widest vertical strip in the middle 35-65% of the text
    width that almost no word crosses (only full-width titles/figure
    captions may). Accepted only if BOTH sides look like prose columns:
    >= 25% of the words each, >= 4 words per line on average (rejects
    term/definition layouts, where the left side is short labels), and
    each side >= 30% of the text width, and each side filled top to
    bottom (see _vertical_fill — rejects borderless two-column tables)."""
    if len(words) < 40:
        return None
    xs0 = min(w["x0"] for w in words)
    xs1 = max(w["x1"] for w in words)
    span = xs1 - xs0
    if span < 200:
        return None

    n_rows = len({round(w["top"]) for w in words})
    lo, hi = int(xs0 + 0.35 * span), int(xs0 + 0.65 * span)
    cover = [0] * (hi - lo + 1)
    for w in words:
        for x in range(max(int(w["x0"]), lo), min(int(w["x1"]) + 1, hi + 1)):
            cover[x - lo] += 1

    allowed = max(2, n_rows // 4)
    best, best_len, start = None, 0, None
    for i, c in enumerate(cover + [allowed + 1]):
        if c <= allowed:
            start = i if start is None else start
        elif start is not None:
            if i - start > best_len:
                best, best_len = (lo + start, lo + i - 1), i - start
            start = None
    if best is None or best_len < 8:
        return None

    gl, gr = best
    mid = (gl + gr) / 2
    side = [w for w in words if not (w["x0"] < gl and w["x1"] > gr)]
    left = [w for w in side if (w["x0"] + w["x1"]) / 2 < mid]
    right = [w for w in side if (w["x0"] + w["x1"]) / 2 >= mid]

    def words_per_line(ws):
        return len(ws) / max(len({round(w["top"]) for w in ws}), 1)

    if len(left) < 0.25 * len(words) or len(right) < 0.25 * len(words):
        return None
    if (max(w["x1"] for w in left) - xs0) < 0.3 * span or (xs1 - min(w["x0"] for w in right)) < 0.3 * span:
        return None
    # Borderless two-column TABLE guard (e.g. "1943 (Early Foundations)" |
    # a 3-line milestone): the shorter-cell side has big vertical holes
    # between rows, while a real text column is filled top to bottom.
    # Each side must cover >= 80% of the height the two sides cover
    # together (so a full-width figure, empty on both sides, doesn't count).
    # Measured: text columns 0.94-1.0; Unit 1 p2's Period|Milestone table
    # 0.63. Failing this only means "treat as single column" = old path.
    two_prose_columns = (
        words_per_line(left) >= 4 and words_per_line(right) >= 4
        and _vertical_fill(left, words) >= 0.8 and _vertical_fill(right, words) >= 0.8
    )
    if two_prose_columns:
        return best
    # OCR'd slide pages only (allow_unaligned=True): also split when one
    # side isn't prose (text on the left, a table/code/figure on the right)
    # as long as the two sides' rows DON'T line up — row order would
    # interleave unrelated lines. Measured on DBMS_U_5_ONE_SHOT: p16 0.0,
    # p33 0.18, p69 0.5; row-aligned layouts (term|definition, top-aligned
    # tables) ~1.0. Not used for text-layer PDFs: there a borderless table
    # with vertically centred cells measured 0.52 (Unit 1 p2 simulation),
    # indistinguishable from p69 — text PDFs keep the prose-only rule.
    if allow_unaligned and _row_alignment(left, right) < 0.6:
        return best
    return None


def _row_alignment(a: List[dict], b: List[dict]) -> float:
    """Share of the side with fewer text rows whose rows have a row on the
    other side at the same height (within 1/4 of a line height)."""
    heights = sorted(w["bottom"] - w["top"] for w in a + b)
    tol = 0.25 * heights[len(heights) // 2]

    def rows(ws):
        out = []
        for t in sorted(w["top"] for w in ws):
            if not out or t - out[-1] > tol:
                out.append(t)
        return out

    ra, rb = rows(a), rows(b)
    if not ra or not rb:
        return 0.0
    small, big = (ra, rb) if len(ra) <= len(rb) else (rb, ra)
    return sum(1 for t in small if any(abs(t - u) <= tol for u in big)) / len(small)


def _merged_spans(ws: List[dict]) -> List[tuple]:
    """Vertical intervals covered by these words, bridging normal line
    spacing and paragraph gaps (<= one median word height)."""
    spans = sorted((w["top"], w["bottom"]) for w in ws)
    if not spans:
        return []
    h = sorted(b - t for t, b in spans)[len(spans) // 2]
    merged = [list(spans[0])]
    for t, b in spans[1:]:
        if t - merged[-1][1] <= h:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([t, b])
    return [tuple(m) for m in merged]


def _vertical_fill(side: List[dict], both: List[dict]) -> float:
    def covered(spans, y0, y1):
        return sum(max(0.0, min(b, y1) - max(t, y0)) for t, b in spans)
    y0 = min(w["top"] for w in side)
    y1 = max(w["bottom"] for w in side)
    union = covered(_merged_spans(both), y0, y1)
    return covered(_merged_spans(side), y0, y1) / union if union else 0.0


def _split_columns(words: List[dict], gutter: tuple) -> List[List[dict]]:
    """[full_width_words, left_words, right_words]. Classified per ROW (not
    per word), so a full-width title isn't cut in three: a row is full-width
    if a word crosses the gutter middle, or if the words on either side of
    it are separated by a normal word gap (< 8pt) rather than a gutter."""
    mid = (gutter[0] + gutter[1]) / 2
    full, left, right = [], [], []
    rows, current, row_top = [], [], None
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if row_top is not None and abs(w["top"] - row_top) > 3.0:
            rows.append(current)
            current = []
        if not current:
            row_top = w["top"]
        current.append(w)
    if current:
        rows.append(current)

    for row in rows:
        row = sorted(row, key=lambda w: w["x0"])
        crosses = any(w["x0"] < mid < w["x1"] for w in row)
        lhs = [w for w in row if (w["x0"] + w["x1"]) / 2 < mid]
        rhs = [w for w in row if (w["x0"] + w["x1"]) / 2 >= mid]
        continuous = bool(lhs and rhs) and (rhs[0]["x0"] - lhs[-1]["x1"]) < 8
        if crosses or continuous:
            full.extend(row)
        else:
            left.extend(lhs)
            right.extend(rhs)
    return [full, left, right]


def _column_reading_order(elements: List[dict], gutter: tuple) -> List[dict]:
    """Band order for a two-column page: every full-width element (title,
    wide figure/table) starts a new band; within a band -> the full-width
    element, then the left column top-to-bottom, then the right column."""
    import bisect
    gl, gr = gutter
    mid = (gl + gr) / 2

    def is_full(b):
        return b[0] < gl and b[2] > gr

    breaks = sorted(e["_bbox"][1] for e in elements if is_full(e["_bbox"]))

    def key(e):
        b = e["_bbox"]
        band = bisect.bisect_right(breaks, b[1])
        col = 0 if is_full(b) else (1 if (b[0] + b[2]) / 2 < mid else 2)
        return (band, col, b[1])

    return sorted(elements, key=key)


def _is_text_box(label_text: str) -> bool:
    """A drawn region whose words are a single line (a slide text box,
    e.g. DBMS p1 "ONE SHOT Revision") or read like prose (> 6 words per
    line on average — Unit 1's bordered paragraphs "Fully Observable
    Environments: The agent has complete ...") is a TEXT BOX, not a
    diagram. Real diagrams are several short labels (sample Figure 1 ~4
    words/line, flowchart test ~1.3). Without this, 18 Unit 1 paragraphs
    became "[Diagram]" and, with vision on, were REPLACED by a 3-6
    sentence summary."""
    lines = [l for l in label_text.split("\n") if l.strip()]
    if len(lines) <= 1:
        return True
    return sum(len(l.split()) for l in lines) / len(lines) > 6


def _find_diagram_regions(page, exclude_bboxes: List[tuple], gap: float = 12.0,
                          min_shapes: int = 3, min_size: float = 40.0) -> List[tuple]:
    """Bounding boxes of DRAWN figures (flowcharts, box-and-arrow diagrams,
    vector charts): clusters of vector shapes (rects / lines / curves) that
    touch or nearly touch each other.

    Replaces _looks_like_diagram's words-per-line guess as the trigger:
    that guess flagged ordinary bullet lists as diagrams (spending vision
    quota on them) and missed real flowcharts whose labels sit in separate
    small boxes. Actual drawn shapes are the reliable signal — prose and
    bullet lists have none. Ignored as non-diagram decoration:
      - any shape spanning >= 80% of the page width/height (page borders,
        notebook ruled lines, margin lines, full-width separators)
      - tiny shapes (< 6pt both ways: bullet dots, specks)
      - shapes inside real tables (already extracted as tables)
      - clusters made only of flat horizontal lines (underlines, rules)
      - clusters too small (< min_size) or covering >= half the page
    """
    W, H = page.width, page.height
    shapes = []  # (x0, top, x1, bottom)
    for o in list(page.rects) + list(page.lines) + list(page.curves):
        b = (o["x0"], o["top"], o["x1"], o["bottom"])
        w, h = b[2] - b[0], b[3] - b[1]
        if w >= 0.8 * W or h >= 0.8 * H:
            continue
        if w < 6 and h < 6:
            continue
        # Padded by 2pt: a table's border lines lie exactly ON its bbox edge,
        # which the strict overlap test missed (each table's frame then
        # became an empty "diagram region" -> a wasted vision call).
        if any(_bboxes_overlap(b, (e[0] - 2, e[1] - 2, e[2] + 2, e[3] + 2)) for e in exclude_bboxes):
            continue
        shapes.append(b)

    # Union-find: shapes whose (gap-padded) boxes touch join one cluster.
    parent = list(range(len(shapes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = shapes[i], shapes[j]
            if (a[0] - gap <= b[2] and b[0] - gap <= a[2]
                    and a[1] - gap <= b[3] and b[1] - gap <= a[3]):
                parent[find(i)] = find(j)

    clusters = {}
    for i in range(len(shapes)):
        clusters.setdefault(find(i), []).append(shapes[i])

    regions = []
    for members in clusters.values():
        if len(members) < min_shapes:
            continue
        if all((m[3] - m[1]) < 2 for m in members):
            continue  # only flat horizontal lines — underlines / rules
        x0 = min(m[0] for m in members); top = min(m[1] for m in members)
        x1 = max(m[2] for m in members); bottom = max(m[3] for m in members)
        if (x1 - x0) < min_size or (bottom - top) < min_size:
            continue
        if (x1 - x0) * (bottom - top) >= 0.5 * W * H:
            continue
        regions.append((x0, top, x1, bottom))
    return regions


def _resolve_diagram_content(page, bbox: tuple, text: str, page_num: int) -> str:
    """Decide what content to store for a diagram/flowchart-looking block.
    Tries a real vision description first (captures structure/relationships,
    not just labels, and works even when word-extraction is garbled) and
    falls back to the existing text-based recovery chain (label text ->
    loose-tolerance reconstruction -> OCR -> placeholder) when vision isn't
    available or fails."""
    image_bytes = _render_region_bytes(page, bbox)
    if image_bytes:
        description = describe_image(image_bytes)
        if description:
            return f"[Diagram] {description}"

    if not is_garbled_text(text):
        return f"[Diagram] {text}"

    reconstructed = _reconstruct_diagram_text(page, bbox)
    if reconstructed and not is_garbled_text(reconstructed):
        return f"[Diagram] {reconstructed}"

    ocr_text = _ocr_region(page, bbox)
    if ocr_text and not is_garbled_text(ocr_text):
        return f"[Diagram] {ocr_text}"

    return (f"[Diagram on page {page_num}: text extraction unreliable, "
            f"{len(text.split())} fragments not shown]")

def _ocr_pdf_pages(path: str, page_numbers: List[int]) -> dict:
    """Full-page OCR fallback for pages with no extractable text at all
    (scanned pages). Returns {page_number: text}, so the caller can insert
    each page's OCR text at THAT page's position in the document instead
    of appending it after every other page."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        return {}
    results = {}
    for n in page_numbers:
        try:
            images = convert_from_path(path, first_page=n, last_page=n)
            text = "\n\n".join(pytesseract.image_to_string(img) for img in images).strip()
            if text:
                results[n] = text
        except Exception:
            continue
    return results


def _image_content_hash(img: dict) -> Optional[str]:
    try:
        import hashlib
        return hashlib.md5(img["stream"].get_rawdata()).hexdigest()
    except Exception:
        return None


def _repeated_image_hashes(pdf) -> set:
    """Image contents that appear on >= 3 pages and >= 30% of all pages —
    watermarks, logos, slide templates. (Compared by content: each page of
    DBMS_U_5_ONE_SHOT.pdf stores its own copy of the same watermark.)"""
    from collections import Counter
    counts = Counter()
    for p in pdf.pages:
        counts.update({h for h in (_image_content_hash(im) for im in p.images) if h})
    n = len(pdf.pages)
    return {h for h, c in counts.items() if c >= 3 and c >= 0.3 * n}


def _is_decorative_image(img: dict, bbox: tuple, page, has_text_layer: bool, repeated: set) -> bool:
    """True for images not worth a vision call: repeated across many pages;
    a (near) full-page background behind a text layer; or small (<10% of the
    page) and touching the page edge (corner/edge logos). Real figures in
    the sample and Unit 1 PDFs are 15-23% of the page and away from edges."""
    W, H = page.width, page.height
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(W * H, 1)
    if area < 0.015:
        # Icons/emoji (TC-U-5 slides: 13 of them on pages 2-4, each <1% of
        # the page) — they'd use up the daily vision budget before the real
        # figures (15-43% of the page in Unit 1 / sample / TC).
        return True
    if repeated and _image_content_hash(img) in repeated:
        return True
    if has_text_layer and area >= 0.9:
        return True
    touches_edge = bbox[0] <= 0.02 * W or bbox[1] <= 0.02 * H or bbox[2] >= 0.98 * W or bbox[3] >= 0.98 * H
    return area < 0.10 and touches_edge


def _transcript_elements(text: str, page_num: int) -> List[dict]:
    """Vision page transcription -> ordered elements: a [TABLE] ... [/TABLE]
    block -> one 'table' element (chunk_elements keeps it whole),
    [Diagram: ...] -> a 'diagram' element, other paragraphs -> 'text'
    (line breaks/indentation kept, e.g. for pseudocode)."""
    els: List[dict] = []
    buf: List[str] = []
    table: Optional[List[str]] = None

    def add(kind: str, content: str):
        els.append({"type": kind, "content": content, "page": page_num, "top": len(els)})

    def flush_text():
        if buf:
            add("text", "\n".join(buf))
            buf.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if table is not None:
            if stripped.upper() == "[/TABLE]":
                if table:
                    add("table", "\n".join(table))
                table = None
            elif stripped:
                table.append(stripped)
            continue
        if stripped.upper() == "[TABLE]":
            flush_text()
            table = []
            continue
        m = re.match(r"^\[Diagram:\s*(.*)$", stripped, re.I)
        if m:
            flush_text()
            add("diagram", "[Diagram] " + m.group(1).rstrip("]").strip())
            continue
        if not stripped:
            flush_text()
            continue
        buf.append(line)
    if table:
        add("table", "\n".join(table))
    flush_text()
    return els


# Private-use code points = symbol-font glyphs (Wingdings/Symbol bullets
# such as U+F0D8 "➢", U+F076 "❖" in PowerPoint exports). They carry no
# text, so they become a plain bullet.
_SYMBOL_FONT_CHARS = re.compile("[\ue000-\uf8ff]")


def _dedupe_overlapping_words(words: List[dict]) -> List[dict]:
    """Text drawn twice with an offset — shadow/outline effects on slide
    titles (TC p1: "A K T U / A K T U", "(Common / (Common") — as a
    separate run, so pdfplumber's dedupe_chars (same position only) can't
    see it. Keep the first copy: drop a word whose box overlaps an earlier
    IDENTICAL word by >= 50% of its area. Genuine repeated words never sit
    on top of each other."""
    kept: List[dict] = []
    by_text: dict = {}
    for w in words:
        area = (w["x1"] - w["x0"]) * (w["bottom"] - w["top"])
        duplicate = False
        for k in by_text.get(w["text"], ()):
            ix = min(w["x1"], k["x1"]) - max(w["x0"], k["x0"])
            iy = min(w["bottom"], k["bottom"]) - max(w["top"], k["top"])
            if area > 0 and ix > 0 and iy > 0 and ix * iy >= 0.5 * area:
                duplicate = True
                break
        if not duplicate:
            kept.append(w)
            by_text.setdefault(w["text"], []).append(w)
    return kept


def _drop_repeated_headers_footers(elements: List[dict], page_heights: dict,
                                   n_pages: int, skip_pages: set) -> List[dict]:
    """Remove running headers/footers: a text/table element in the top or
    bottom 15% of its page whose content (letters+digits) repeats on >= 30%
    of pages (and >= 3). Measured on TC-U-5-ONE-SHOT-Notes.pdf: the footer
    "GateWay Classes Helpline Number ..." / ": Follow Us" (the footer's boxes
    were even detected as a table on 56 pages) and the header
    "Technical Communication / Unit-5 : One Shot" were 14% of all words.
    Short elements only (<= 15 words); pages from vision transcription are
    skipped (their 'top' is an order index, not a position)."""
    from collections import defaultdict

    def key(e):
        return re.sub(r"[^a-z0-9]", "", e["content"].lower())

    def in_band(e):
        h = page_heights.get(e["page"])
        return bool(h) and (e["top"] < 0.15 * h or e["top"] > 0.85 * h)

    def candidate(e):
        return (e["type"] in ("text", "table") and e["page"] not in skip_pages
                and len(e["content"].split()) <= 15 and in_band(e) and key(e))

    pages_by_key = defaultdict(set)
    for e in elements:
        if candidate(e):
            pages_by_key[key(e)].add(e["page"])
    need = max(3, 0.3 * n_pages)
    repeated = {k for k, pages in pages_by_key.items() if len(pages) >= need}
    if not repeated:
        return elements
    return [e for e in elements if not (candidate(e) and key(e) in repeated)]


def _load_pdf(path: str) -> List[dict]:
    try:
        import pdfplumber
    except ImportError:
        return _load_pdf_pypdf_fallback(path)

    elements: List[dict] = []
    scanned_pages: List[int] = []

    with pdfplumber.open(path) as pdf:
        # Pages with NO text layer but with an image (slides/scans): read
        # them with local OCR first (see page_ocr.py). Done as one pass over
        # the document so repeated logos/headers can be recognized and
        # dropped. {} if rapidocr isn't installed -> old behaviour.
        image_only = []
        for n, p in enumerate(pdf.pages, 1):
            try:
                has_text = bool(p.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False))
            except Exception:
                has_text = False
            if not has_text and p.images:
                image_only.append(n)
        ocr_words_by_page, vision_reasons = ocr_pages(path, image_only)
        # Pages OCR can't do justice (nothing/mostly unreadable, grid
        # tables) -> ONE batched vision transcription for all of them.
        # Missing pages (no key, quota, budget, failure) keep their OCR.
        vision_text_by_page = transcribe_pages_with_vision(path, sorted(vision_reasons))
        # Image contents repeated across many pages (watermarks, logos,
        # slide templates) — only needed for the vision image loop below.
        images_readable = USE_REAL_VISION or ocr_available()
        repeated_images = _repeated_image_hashes(pdf) if images_readable else set()
        page_heights = {n: p.height for n, p in enumerate(pdf.pages, 1)}

        for page_num, page in enumerate(pdf.pages, 1):
            page_elements = []

            if page_num in vision_text_by_page:
                elements.extend(_transcript_elements(vision_text_by_page[page_num], page_num))
                continue

            # "Fake bold" (each glyph printed twice, slightly offset) came out
            # as "CCoommmmoonn" / duplicated lines (DBMS p1, TC p1).
            try:
                page = page.dedupe_chars()
            except Exception:
                pass

            real_tables = _find_real_tables(page)
            table_bboxes = [t["bbox"] for t in real_tables]
            for t in real_tables:
                table_text = _table_to_text(t["rows"])
                if table_text:
                    page_elements.append({
                        "type": "table", "content": table_text,
                        "page": page_num, "top": t["bbox"][1], "_bbox": t["bbox"],
                    })

            try:
                words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)
            except Exception:
                words = []
            ocr_page = not words and page_num in ocr_words_by_page
            if ocr_page:
                words = ocr_words_by_page[page_num]
            else:
                words = _dedupe_overlapping_words(words)
            # Drop words that fall inside a real table — already captured,
            # verbalized, by the table extraction above. Avoids the same
            # content appearing twice, once as a table row and once as
            # loose paragraph text.
            non_table_words = [w for w in words if not _word_in_bbox(w, table_bboxes)]

            diagram_bboxes = []  # for skipping embedded images already covered below

            # Drawn figures: every word inside a region becomes part of ONE
            # diagram element (vision-described when available, its labels
            # otherwise), instead of loose one-line text blocks.
            diagram_regions = _find_diagram_regions(page, table_bboxes)
            if diagram_regions:
                region_words = {r: [] for r in diagram_regions}
                loose_words = []
                for w in non_table_words:
                    owner = next((r for r in diagram_regions if _word_in_bbox(w, [r])), None)
                    (region_words[owner] if owner else loose_words).append(w)
                non_table_words = loose_words
                for region in diagram_regions:
                    label_text = "\n".join(
                        b["text"] for b in _group_words_into_blocks(region_words[region])
                    ).strip()
                    region_area = (region[2] - region[0]) * (region[3] - region[1])
                    if not label_text and region_area < 0.05 * page.width * page.height:
                        continue  # small text-less shapes (icon, logo frame): not a figure
                    if label_text and _is_text_box(label_text):
                        # A bordered paragraph/callout, not a diagram: keep
                        # its words as normal text (verbatim, no vision call).
                        non_table_words.extend(region_words[region])
                        continue
                    content = _resolve_diagram_content(page, region, label_text, page_num)
                    if content.strip() == "[Diagram]":
                        continue  # no labels and no vision description — nothing to keep
                    elem_type = "diagram_placeholder" if content.startswith("[Diagram on page") else "diagram"
                    page_elements.append({
                        "type": elem_type, "content": content,
                        "page": page_num, "top": region[1], "_bbox": region,
                    })
                    diagram_bboxes.append(region)

            # Two-column pages: group each column's words into lines/blocks
            # separately (otherwise every row glues a left-column line to
            # the right-column line beside it). None -> unchanged path.
            gutter = _find_column_gutter(non_table_words, allow_unaligned=ocr_page)
            if gutter:
                blocks = [b for group in _split_columns(non_table_words, gutter)
                          for b in _group_words_into_blocks(group)]
            else:
                blocks = _group_words_into_blocks(non_table_words)

            for block in blocks:
                text = block["text"].strip()
                if not text:
                    continue

                bbox = (block["x0"], block["top"], block["x1"], block["bottom"])

                # Garbled extraction (almost always a diagram whose labels
                # got chopped into fragments) routes this block through
                # vision-first description + the text-based fallback chain
                # instead of treating it as plain prose. (Drawn diagrams are
                # already handled above via _find_diagram_regions; the old
                # _looks_like_diagram layout guess is no longer a trigger —
                # it flagged ordinary bullet lists.)
                # (Not for OCR pages: OCR output isn't the pdfplumber
                # char-fragmentation this check targets.)
                if not ocr_page and is_garbled_text(text):
                    content = _resolve_diagram_content(page, bbox, text, page_num)
                    elem_type = "diagram_placeholder" if content.startswith("[Diagram on page") else "diagram"
                    page_elements.append({
                        "type": elem_type, "content": content,
                        "page": page_num, "top": block["top"], "_bbox": bbox,
                    })
                    diagram_bboxes.append(bbox)
                    continue

                page_elements.append({
                    "type": "text", "content": text,
                    "page": page_num, "top": block["top"], "_bbox": bbox,
                })

            # Standalone embedded images (photos, charts, screenshots) —
            # invisible to the words-based extraction above since they
            # carry no text glyphs at all. Only attempted when vision
            # description is available; without it there's nothing
            # meaningful to pull from a pure image, so skip rather than
            # emit an empty/junk element.
            if images_readable and not ocr_page:
             excluded_bboxes = table_bboxes + diagram_bboxes
             for img in page.images:
                 img_bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                 if any(_bboxes_overlap(img_bbox, b) for b in excluded_bboxes):
                     continue
                 # Skip tiny images (icons, bullets, logos) — not worth
                 # a vision call and rarely meaningful content.
                 if (img_bbox[2] - img_bbox[0]) < 40 or (img_bbox[3] - img_bbox[1]) < 40:
                     continue
                 # Skip decoration (backgrounds, logos, watermarks) — each
                 # would otherwise cost a vision call for nothing.
                 if _is_decorative_image(img, img_bbox, page, bool(words), repeated_images):
                     continue
 
                 description = ""
                 if USE_REAL_VISION:
                     image_bytes = _render_region_bytes(page, img_bbox)
                     if image_bytes:
                         description = describe_image(image_bytes)
 
                 if description:
                     page_elements.append({
                         "type": "image", "content": f"[Image] {description}",
                         "page": page_num, "top": img_bbox[1], "_bbox": img_bbox,
                     })
                     continue
 
                 # No vision description available (no key set, or the call
                 # failed) — fall back to OCR in case the image itself has
                 # text baked into the pixels (e.g. chart labels). A pure
                 # photo with no text still won't recover anything here —
                 # that's a real limitation without a vision key.
                 ocr_text = _ocr_region(page, img_bbox)
                 if ocr_text and len(ocr_text.split()) >= 3 and not is_garbled_text(ocr_text):
                     page_elements.append({
                         "type": "image", "content": f"[Image, OCR only] {ocr_text}",
                         "page": page_num, "top": img_bbox[1], "_bbox": img_bbox,
                     })

            if not page_elements:
                scanned_pages.append(page_num)
            else:
                if gutter:
                    page_elements = _column_reading_order(page_elements, gutter)
                else:
                    page_elements.sort(key=lambda e: e["top"])
                for e in page_elements:
                    e.pop("_bbox", None)
                elements.extend(page_elements)

        elements = _drop_repeated_headers_footers(
            elements, page_heights, len(pdf.pages), set(vision_text_by_page))

    if scanned_pages:
        # Full-page OCR fallback, inserted AT each page's own position in
        # the element list (not appended after every other page — that was
        # the old bug: OCR text always landed at the very end of the
        # document regardless of which page it actually came from).
        ocr_by_page = _ocr_pdf_pages(path, scanned_pages)
        still_missing = []
        for page_num in scanned_pages:
            text = ocr_by_page.get(page_num)
            if text:
                insert_at = next((i for i, e in enumerate(elements) if e["page"] > page_num), len(elements))
                elements.insert(insert_at, {
                    "type": "text", "content": text, "page": page_num, "top": 0,
                })
            else:
                still_missing.append(page_num)
        if still_missing:
            print(f"(pdf: pages {still_missing} appear to be scanned/image-only and "
                  "OCR isn't available — install pytesseract + pdf2image + poppler "
                  "to recover their text)")

    for e in elements:
        e["content"] = _SYMBOL_FONT_CHARS.sub("•", e["content"])
    return elements


def _load_pdf_pypdf_fallback(path: str) -> List[dict]:
    print("(pdfplumber not installed — falling back to pypdf; tables and "
          "diagrams won't be distinguished from plain text. "
          "`pip install pdfplumber` for better results)")
    from pypdf import PdfReader
    reader = PdfReader(path)
    elements = []
    for page_num, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            elements.append({"type": "text", "content": text, "page": page_num, "top": 0})
    return elements


def _table_to_text(table: List[List[str]]) -> str:
    """Verbalize a table row-by-row (header: value, header: value) rather
    than leaving raw grid cells — embeds/retrieves far better than a
    pipe-delimited grid, since the model matches on natural phrasing."""
    if not table or not table[0]:
        return ""
    header = [str(h or "").strip() for h in table[0]]
    rows_text = []
    for row in table[1:]:
        cells = [str(c or "").strip() for c in row]
        pairs = [f"{h}: {c}" for h, c in zip(header, cells) if c]
        if pairs:
            rows_text.append("; ".join(pairs))
    return "\n".join(rows_text)


# ---------------------------------------------------------------------------
# PowerPoint — includes speaker notes
# ---------------------------------------------------------------------------

def _load_pptx(path: str) -> str:
    from pptx import Presentation
    prs = Presentation(path)
    slide_texts = []
    for i, slide in enumerate(prs.slides, 1):
        parts = [f"[Slide {i}]"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[Speaker notes] {notes}")
        slide_texts.append("\n".join(parts))
    return "\n\n".join(slide_texts)


# ---------------------------------------------------------------------------
# Word documents
# ---------------------------------------------------------------------------

def _load_docx(path: str) -> str:
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx is required for .docx files. "
                            "Install with: pip install python-docx")
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if rows:
            parts.append(_table_to_text(rows))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tabular data — CSV / Excel, verbalized row-by-row
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> str:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required for .csv files. "
                            "Install with: pip install pandas")
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")  # Excel-on-Windows exports
    return _dataframe_to_text(df, os.path.basename(path))


def _load_excel(path: str) -> str:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas + openpyxl are required for .xlsx/.xls "
                            "files. Install with: pip install pandas openpyxl")
    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(_dataframe_to_text(df, f"{os.path.basename(path)} — {sheet_name}"))
    return "\n\n".join(parts)


def _dataframe_to_text(df, source_label: str, max_rows: int = 2000) -> str:
    parts = [f"[{source_label}] columns: {', '.join(str(c) for c in df.columns)}"]
    for _, row in df.head(max_rows).iterrows():
        pairs = [f"{col}: {row[col]}" for col in df.columns if str(row[col]) != "nan"]
        parts.append("; ".join(pairs))
    if len(df) > max_rows:
        parts.append(f"[...{len(df) - max_rows} more rows truncated...]")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _load_html(path: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("beautifulsoup4 is required for .html files. "
                            "Install with: pip install beautifulsoup4")
    soup = BeautifulSoup(_read_text_file(path), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# JSON — flattened to readable key: value lines
# ---------------------------------------------------------------------------

def _load_json(path: str) -> str:
    data = _json.loads(_read_text_file(path))
    return _flatten_json(data)


def _flatten_json(obj, prefix: str = "") -> str:
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            lines.append(_flatten_json(v, key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            lines.append(_flatten_json(item, f"{prefix}[{i}]"))
    else:
        return f"{prefix}: {obj}"
    return "\n".join(l for l in lines if l)


# ---------------------------------------------------------------------------
# Images — OCR
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _load_image(path: str) -> str:
    """Flat text of an image file (see _load_image_structured)."""
    return "\n\n".join(e["content"] for e in _load_image_structured(path))


def _load_image_structured(path: str) -> List[dict]:
    """A standalone image as the source (photo of notes, slide screenshot,
    diagram) gets the same treatment as an image-only PDF page: EXIF
    rotation fixed (phone photos are often stored sideways), local OCR with
    column/reading-order layout, and ONE vision transcription when OCR
    can't do it justice (nothing read, mostly unreadable/handwriting, grid
    table). Tesseract is only a last resort. Before this, image files went
    straight to Tesseract (not installed here -> error) with no layout,
    tables or vision."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

    words, vision_reason = ocr_image_words(img)
    if vision_reason and USE_REAL_VISION:
        text = transcribe_pages([img])[0]   # '' if unavailable (counted as a shortfall)
        if text.strip():
            return _transcript_elements(text, None)
    if words:
        return _elements_from_words(words)

    try:
        import pytesseract
        text = pytesseract.image_to_string(img)
        if text.strip():
            return [{"type": "text", "content": u, "page": None, "top": i}
                    for i, u in enumerate(_split_into_units(text))]
    except Exception:
        pass
    if not ocr_available() and not USE_REAL_VISION:
        raise RuntimeError(
            "Reading text from an image needs local OCR (pip install rapidocr_onnxruntime) "
            "or a GOOGLE_API_KEY for vision.")
    print(f"(image: no readable text found in {os.path.basename(path)})")
    return []


def _elements_from_words(words: List[dict], page_num=None) -> List[dict]:
    """OCR words of one image -> ordered text elements, with the same
    two-column handling as OCR'd PDF pages (allow_unaligned=True)."""
    gutter = _find_column_gutter(words, allow_unaligned=True)
    groups = _split_columns(words, gutter) if gutter else [words]
    els = []
    for group in groups:
        for b in _group_words_into_blocks(group):
            text = b["text"].strip()
            if text:
                els.append({"type": "text", "content": text, "page": page_num, "top": b["top"],
                            "_bbox": (b["x0"], b["top"], b["x1"], b["bottom"])})
    els = _column_reading_order(els, gutter) if gutter else sorted(els, key=lambda e: e["top"])
    for e in els:
        e.pop("_bbox", None)
    return els


LOADERS = {
    ".md": _load_text,
    ".txt": _load_text,
    ".pptx": _load_pptx,
    ".docx": _load_docx,
    ".csv": _load_csv,
    ".xlsx": _load_excel,
    ".xls": _load_excel,
    ".html": _load_html,
    ".htm": _load_html,
    ".json": _load_json,
    ".png": _load_image,
    ".jpg": _load_image,
    ".jpeg": _load_image,
    ".webp": _load_image,
    ".bmp": _load_image,
    ".tif": _load_image,
    ".tiff": _load_image,
}
# ".pdf" is handled specially in load_document / load_document_structured
# (it returns structured elements, not a plain string).


# ---------------------------------------------------------------------------
# Chunking — packs natural structural units instead of blind word counts
# ---------------------------------------------------------------------------

def _split_into_units(text: str) -> List[str]:
    """Split text into natural structural units — paragraphs, table blocks,
    slide blocks — so units can be packed whole into a chunk instead of
    being blindly cut at a word count."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    units = []
    for block in blocks:
        if len(block.split()) <= 60:
            units.append(block)
        else:
            units.extend(l for l in block.split("\n") if l.strip())
    return units


def chunk_elements(elements: List[dict], chunk_size: int = 200, overlap: int = 40) -> List[dict]:
    """Structure-aware chunking: works on ordered (type, content, page)
    elements instead of a flat string, so a table row or a diagram
    description is packed as one atomic unit — never sliced mid-row or
    mid-diagram — and each output chunk carries metadata (chunk_id,
    page(s), element types present, position) for neighbor/section
    expansion at retrieval time.
    """
    units = []  # (text, page, type)
    for el in elements:
        if el["type"] in ("table", "diagram", "diagram_placeholder", "image"):
            units.append((el["content"], el.get("page"), el["type"]))
        else:
            for u in _split_into_units(el["content"]):
                units.append((u, el.get("page"), "text"))

    if not units:
        return []

    chunks: List[dict] = []
    current: List[tuple] = []
    current_words = 0

    def flush():
        if not current:
            return
        text = "\n".join(u[0] for u in current)
        pages = sorted({u[1] for u in current if u[1] is not None})
        types = sorted({u[2] for u in current})
        chunks.append({
            "chunk_id": f"c{len(chunks)}",
            "text": text,
            "page": pages[0] if len(pages) == 1 else (pages or None),
            "types": types,
            "position": len(chunks),
        })

    i = 0
    while i < len(units):
        text, page, utype = units[i]
        unit_words = len(text.split())

        if unit_words > chunk_size and utype == "text":
            # Only plain prose gets force-split by raw word count; tables
            # and diagrams stay atomic even if that makes one chunk larger
            # than chunk_size — splitting a table row or a diagram
            # description produces meaningless fragments.
            flush()
            current, current_words = [], 0
            words = text.split()
            step = max(chunk_size - overlap, 1)
            for start in range(0, len(words), step):
                piece = words[start:start + chunk_size]
                if not piece:
                    break
                chunks.append({
                    "chunk_id": f"c{len(chunks)}",
                    "text": " ".join(piece),
                    "page": page, "types": ["text"], "position": len(chunks),
                })
                if start + chunk_size >= len(words):
                    break
            i += 1
            continue

        if current_words + unit_words > chunk_size and current:
            flush()
            carried, carried_words = [], 0
            for prev in reversed(current):
                w = len(prev[0].split())
                if carried_words + w > overlap:
                    break
                carried.insert(0, prev)
                carried_words += w
            current, current_words = carried, carried_words

        current.append((text, page, utype))
        current_words += unit_words
        i += 1

    flush()
    return chunks


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """Back-compat convenience: chunk a plain string with no table/diagram/
    page metadata. New code should prefer load_document_structured() +
    chunk_elements() to get that metadata."""
    elements = [{"type": "text", "content": text, "page": None}]
    return [c["text"] for c in chunk_elements(elements, chunk_size=chunk_size, overlap=overlap)]