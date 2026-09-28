"""
rag/page_ocr.py — local OCR for PDF pages that have NO text layer (slides
exported as images, lecture-video frames, scans).

Why: a PDF like DBMS_U_5_ONE_SHOT.pdf is 71 full-page screenshots — no
extractable text at all. Before this, those pages produced nothing (the
Tesseract/Poppler fallback isn't installed) or burned one Gemini vision
call per image (20/day) with a "describe this diagram" summary instead of
the slide's actual text.

Engine: RapidOCR (PaddleOCR models on ONNX Runtime) — pip-only, free,
offline, ~1.5-2.5s per slide on CPU. Optional: if it isn't installed,
ocr_image_only_pages() returns {} and ingestion behaves exactly as before.

What it returns is shaped like pdfplumber's extract_words() output
({"text","x0","x1","top","bottom"} in PDF points), so the page then goes
through the SAME layout code as a normal text page (two-column detection,
block grouping, reading order).

Measured on DBMS_U_5_ONE_SHOT.pdf (35 hand-checked lines):
  - The Chinese+English model drops most English spaces
    ("Twooperations,lock_item andunlock_item"). Restored by
    _restore_spaces(): a wordninja dictionary boundary between two letters
    that ALSO has a small visible gap (>= 0.20 x median char width) in
    RapidOCR's per-character boxes. Word accuracy: raw 50% -> 98%
    (gap-only 77%, dictionary-only 95%).
  - Lines below 0.8 confidence are dropped (handwriting came out as
    "Conseahve 2h" at ~0.7) — such pages stay unread and fall through to
    vision / the coverage warning instead of feeding junk.
  - Short lines repeating on >= 30% of OCR'd pages (logos, "GW",
    "GATEWAY CLASSES") are dropped as boilerplate.
Results are cached per page image in .ocr_cache/ (the quiz re-reads the
whole PDF; without the cache that would redo minutes of OCR).
"""

import difflib
import hashlib
import json
import os
import re
import statistics
from typing import Dict, List, Optional

_OCR_VERSION = "rapidocr-v1"
_OCR_CACHE_DIR = os.environ.get("OCR_CACHE_DIR", ".ocr_cache")
MIN_LINE_CONFIDENCE = 0.8
DOMINANT_IMAGE_AREA = 0.8      # one image covering >= 80% of the page -> OCR it at native resolution
RENDER_DPI = 200               # otherwise render the whole page at this DPI
BOILERPLATE_PAGE_SHARE = 0.3
LETTER_GAP_RATIO = 0.20
OTHER_GAP_RATIO = 0.55

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        except Exception:
            _engine = False
    return _engine or None


def ocr_available() -> bool:
    return _get_engine() is not None


# ---------------------------------------------------------------------------
# Page image
# ---------------------------------------------------------------------------

def _page_image(pdfium_page):
    """(PIL image, (x0, top, x1, bottom) in PDF points the image covers)."""
    import pypdfium2 as pdfium
    W, H = pdfium_page.get_size()
    try:
        imgs = [o for o in pdfium_page.get_objects() if o.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE]
        if imgs:
            best = max(imgs, key=lambda o: o.get_px_size()[0] * o.get_px_size()[1])
            left, bottom, right, top = best.get_pos()
            if (right - left) * (top - bottom) >= DOMINANT_IMAGE_AREA * W * H:
                img = best.get_bitmap(render=False).to_pil().convert("RGB")
                return img, (left, H - top, right, H - bottom)
    except Exception:
        pass  # unusual image encoding — fall back to rendering the page
    img = pdfium_page.render(scale=RENDER_DPI / 72).to_pil().convert("RGB")
    return img, (0.0, 0.0, W, H)


# ---------------------------------------------------------------------------
# OCR + cache
# ---------------------------------------------------------------------------

def _ocr_lines(img) -> List[dict]:
    """[{"box": [x0, top, x1, bottom] px, "score", "chars": [[c, x0, x1], ...]}]"""
    import numpy as np
    key = hashlib.md5(img.tobytes() + _OCR_VERSION.encode()).hexdigest()
    cache_path = os.path.join(_OCR_CACHE_DIR, f"{key}.json")
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    result, _ = _get_engine()(np.asarray(img), return_word_box=True)
    lines = []
    for item in result or []:
        box, text, score = item[0], item[1], float(item[2])
        char_boxes, chars = (item[3], item[4]) if len(item) >= 5 else ([], list(text))
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        lines.append({
            "box": [min(xs), min(ys), max(xs), max(ys)],
            "score": score,
            "chars": [[c, float(b[0][0]), float(b[1][0])] for c, b in zip(chars, char_boxes)]
                     or [[c, min(xs), max(xs)] for c in text],
        })
    os.makedirs(_OCR_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(lines, f)
    return lines


# ---------------------------------------------------------------------------
# Space restoration -> words with x-ranges
# ---------------------------------------------------------------------------

def _dictionary_boundaries(text: str) -> set:
    try:
        import wordninja
    except ImportError:
        return set()
    b = set()
    for m in re.finditer(r"[A-Za-z]+", text):
        pos = m.start()
        for piece in wordninja.split(m.group())[:-1]:
            pos += len(piece)
            b.add(pos)
    return b


def _words_from_line(chars: List[list]) -> List[tuple]:
    """[(word_text, x0_px, x1_px)] with dropped spaces restored."""
    text = "".join(c for c, _, _ in chars)
    widths = [x1 - x0 for c, x0, x1 in chars if c != " "]
    median_w = statistics.median(widths) if widths else 1.0
    bounds = _dictionary_boundaries(text)
    words, cur = [], []
    for i, (c, x0, x1) in enumerate(chars):
        if c == " ":
            if cur:
                words.append(cur); cur = []
            continue
        if cur:
            gap = x0 - chars[i - 1][2]
            prev = chars[i - 1][0]
            if prev.isalpha() and c.isalpha():
                split = i in bounds and gap >= LETTER_GAP_RATIO * median_w
            else:
                split = gap >= OTHER_GAP_RATIO * median_w
            if split:
                words.append(cur); cur = []
        cur.append((c, x0, x1))
    if cur:
        words.append(cur)
    return [("".join(c for c, _, _ in w), w[0][1], w[-1][2]) for w in words]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _boilerplate_keys(lines_by_page: Dict[int, List[tuple]]) -> List[str]:
    """Keys of short lines (<= 3 words) that repeat, fuzzily, on >= 30% of
    the OCR'd pages (and at least 3): logos, channel names, running headers."""
    n_pages = len(lines_by_page)
    if n_pages < 3:
        return []
    clusters: List[list] = []  # [representative_key, set(pages)]
    for pn, lines in lines_by_page.items():
        for words, _ in lines:
            if len(words) > 3:
                continue
            k = _key(" ".join(w for w, _, _ in words))
            if len(k) < 2:
                continue
            for cl in clusters:
                if difflib.SequenceMatcher(None, k, cl[0]).ratio() >= 0.75:
                    cl[1].add(pn); break
            else:
                clusters.append([k, {pn}])
    need = max(3, BOILERPLATE_PAGE_SHARE * n_pages)
    return [k for k, pages in clusters if len(pages) >= need]


def _is_boilerplate(words, keys) -> bool:
    if len(words) > 3 or not keys:
        return False
    k = _key(" ".join(w for w, _, _ in words))
    return any(difflib.SequenceMatcher(None, k, b).ratio() >= 0.75 for b in keys)


# ---------------------------------------------------------------------------
# Which OCR'd pages should go to vision instead (see transcribe_pages)
# ---------------------------------------------------------------------------

UNREADABLE_SHARE = 0.3     # >= 30% of detected lines below MIN_LINE_CONFIDENCE
TABLE_COL_TOLERANCE = 0.01 # cell centres within 1% of the image width...
TABLE_MIN_ROWS = 4         # ...in >= 4 rows, for >= 3 columns = a real grid table


def _grid_table_columns(kept_lines: List[dict], width: float) -> int:
    """Number of columns whose cells line up (centres within 1% of the
    width) across >= 4 rows of short cells (<= 3 words, >= 3 cells a row).
    Measured on DBMS_U_5_ONE_SHOT.pdf: the p69 lock matrix -> 6; justified
    two-column prose (whose OCR word boxes look like short cells) -> <= 1.
    A looser rule (2% / 3 rows) wrongly flagged 6 prose pages."""
    rows = {}
    for line in kept_lines:
        if len(_words_from_line(line["chars"])) <= 3:
            rows.setdefault(round(line["box"][1] / 25), []).append((line["box"][0] + line["box"][2]) / 2)
    cand = [xs for xs in rows.values() if len(xs) >= 3]
    tol = TABLE_COL_TOLERANCE * width
    clusters = []
    for x in sorted(x for xs in cand for x in xs):
        if clusters and x - clusters[-1][-1] <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return sum(1 for c in clusters
               if sum(1 for xs in cand if any(abs(x - sum(c) / len(c)) <= tol for x in xs)) >= TABLE_MIN_ROWS)


def _vision_reason(all_lines: List[dict], kept_lines: List[dict], width: float) -> Optional[str]:
    if not kept_lines:
        return "OCR read nothing"
    dropped = len(all_lines) - len(kept_lines)
    if dropped / len(all_lines) >= UNREADABLE_SHARE:
        return f"{dropped}/{len(all_lines)} lines unreadable (handwriting/figure)"
    if _grid_table_columns(kept_lines, width) >= 3:
        return "grid table"
    return None


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def ocr_image_only_pages(path: str, page_numbers: List[int]) -> Dict[int, List[dict]]:
    """{page_number: pdfplumber-style word dicts} for the given pages (those
    with no text layer). Pages where OCR finds nothing usable are omitted."""
    return ocr_pages(path, page_numbers)[0]


def ocr_pages(path: str, page_numbers: List[int]):
    """(words_by_page, vision_reasons). words_by_page as for
    ocr_image_only_pages(); vision_reasons = {page_number: why} for pages
    OCR can't do justice (nothing read, mostly unreadable lines, or a grid
    table whose structure OCR flattens) — ingest sends those to
    transcribe_pages_with_vision() and keeps the OCR words as fallback."""
    if not page_numbers or not ocr_available():
        return {}, {}
    import pypdfium2 as pdfium

    lines_by_page: Dict[int, List[tuple]] = {}   # pn -> [(words_px, (top, bottom))]
    geometry = {}                                 # pn -> (sx, sy, x_off, y_off)
    vision_reasons: Dict[int, str] = {}
    doc = pdfium.PdfDocument(path)
    try:
        for pn in page_numbers:
            img, (bx0, btop, bx1, bbottom) = _page_image(doc[pn - 1])
            geometry[pn] = ((bx1 - bx0) / img.width, (bbottom - btop) / img.height, bx0, btop)
            all_lines = _ocr_lines(img)
            kept_lines = [l for l in all_lines if l["score"] >= MIN_LINE_CONFIDENCE]
            reason = _vision_reason(all_lines, kept_lines, img.width) if all_lines else "OCR read nothing"
            if reason:
                vision_reasons[pn] = reason
            kept = []
            for line in kept_lines:
                words = _words_from_line(line["chars"])
                if words:
                    kept.append((words, (line["box"][1], line["box"][3])))
            if kept:
                lines_by_page[pn] = kept
    finally:
        doc.close()

    boiler = _boilerplate_keys(lines_by_page)
    out: Dict[int, List[dict]] = {}
    for pn, lines in lines_by_page.items():
        sx, sy, ox, oy = geometry[pn]
        page_words = []
        for words, (top, bottom) in lines:
            if _is_boilerplate(words, boiler):
                continue
            for text, x0, x1 in words:
                page_words.append({
                    "text": text,
                    "x0": ox + x0 * sx, "x1": ox + x1 * sx,
                    "top": oy + top * sy, "bottom": oy + bottom * sy,
                })
        if page_words:
            out[pn] = page_words
    return out, vision_reasons


def ocr_image_words(img, longest_side: float = 1000.0):
    """(words, vision_reason) for a STANDALONE image file — the same OCR,
    space restoration, confidence filter and "does vision do better?" check
    as an image-only PDF page. Word boxes are scaled so the image's longest
    side is ~1000 units (about a slide page in points), so the layout
    code's absolute thresholds (line/block gaps, gutter width) behave the
    same whatever the photo's resolution. ([], reason) if OCR is missing."""
    if not ocr_available():
        return [], "OCR not available"
    img = img.convert("RGB")
    all_lines = _ocr_lines(img)
    kept = [l for l in all_lines if l["score"] >= MIN_LINE_CONFIDENCE]
    reason = _vision_reason(all_lines, kept, img.width) if all_lines else "OCR read nothing"
    s = longest_side / max(img.width, img.height)
    words = []
    for line in kept:
        top, bottom = line["box"][1] * s, line["box"][3] * s
        for text, x0, x1 in _words_from_line(line["chars"]):
            words.append({"text": text, "x0": x0 * s, "x1": x1 * s, "top": top, "bottom": bottom})
    return words, reason


def ocr_image_text(img) -> str:
    """Plain text of an image REGION (e.g. a figure on a normal text page)
    via RapidOCR: confident lines only, spaces restored, top-to-bottom then
    left-to-right. '' if OCR isn't available. Used by ingest._ocr_region
    as the offline fallback when vision isn't available (Tesseract, the old
    fallback, isn't installed)."""
    if not ocr_available():
        return ""
    img = img.convert("RGB")
    lines = [l for l in _ocr_lines(img) if l["score"] >= MIN_LINE_CONFIDENCE]
    lines.sort(key=lambda l: (round(l["box"][1] / 20), l["box"][0]))
    out = [" ".join(w for w, _, _ in _words_from_line(l["chars"])) for l in lines]
    return "\n".join(t for t in out if t.strip())


def transcribe_pages_with_vision(path: str, page_numbers: List[int]) -> Dict[int, str]:
    """{page_number: verbatim transcription} via Gemini vision, batched
    (vision_describe.transcribe_pages). Pages vision couldn't do (no key,
    quota/overload block, daily budget, failure) are omitted."""
    if not page_numbers:
        return {}
    from .vision_describe import transcribe_pages, USE_REAL_VISION
    if not USE_REAL_VISION:
        return {}
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    try:
        images = [_page_image(doc[pn - 1])[0] for pn in page_numbers]
    finally:
        doc.close()
    texts = transcribe_pages(images)
    return {pn: t for pn, t in zip(page_numbers, texts) if t.strip()}
