# CLAUDE.md — Teacher_student_model (AI Teaching Classroom)

Context for Claude Code. Read this first; it captures history and decisions
that aren't obvious from the code alone.

## What this project is
A voice-based "live class" for one learner (Kaustubh). A PhD-level teacher
agent teaches from a source (PDF / docx / pptx / md / txt / csv / xlsx /
html / json / image / **YouTube URL**) or a bare topic. The student can
interrupt by voice at any moment; the teacher answers the doubt and
resumes the same section. After the lecture a quiz is generated.
Comparable to NotebookLM, but interactive.

Long-term vision (NOT built yet, intentionally LAST): a "curious student"
agent — a peer who already studied the material and acts as a
coverage/fidelity watchdog (flags skipped topics or drift). Must be
infrequent and well-timed so it never irritates the real student.

## How to run
```
python classroom.py <document_path_or_youtube_url> <student_name> ["optional topic override"]
```
- Windows machine, Python 3.11, venv in `.venv/`.
- Keys in `.env`: `GROQ_API_KEY` (teacher/planner), `GOOGLE_API_KEY`
  (quiz + vision). Without a key each part falls back to an offline stub.
- NEVER print, edit, or commit `.env`.
- `README.md` is outdated (describes Phase 0 `teacher.py` only).

## Architecture (main entry: classroom.py — LangGraph StateGraph)
`plan_lecture` → `teach_and_listen` (loops once per section) → `generate_quiz_node` → END

- **planner.py** — Planner agent (Groq `openai/gpt-oss-20b`). Samples the
  document evenly (stride sampling, 2500-word cap) to propose 4–7 section
  titles, then `assign_context` pre-retrieves chunks per section
  (k=5, min_score 0.15, cross-section dedup, neighbor expansion).
  `_parse_outline` is robust to ```json fences / preambles / bullet lists;
  `MIN_SECTIONS=4`, `MAX_SECTIONS=7` enforced in code, one strict retry.
- **explainer.py** — Explainer (Groq `openai/gpt-oss-120b`). Teaches each
  section in 3 "beats" (introduction → deep_dive → connection) as separate
  LLM calls to stay under Groq's 8000 TPM free-tier limit. Beats go through
  `teacher.call_llm` (its own gpt-oss-120b client) wrapped in
  `_call_llm_with_rate_limit_retry` (on a Groq 429: wait Groq's suggested
  delay, max 2 retries; waits > 65s e.g. per-day limits are raised).
  `answer_doubt` uses explainer's own ChatGroq client, can call tools
  `search_document` / `search_web` (tools.py), and replays up to 6 prior
  doubts from this lecture.
  `BeatStream` (since 2026-09-27): classroom.teach_and_listen generates a
  section's beats in a background thread and starts speaking beat 1 as
  soon as it's ready, so Groq TPM waits on beats 2-3 overlap with speech.
  Measured on DBMS PDF: launch → first section sentence 63s → ~21s (beat 3
  still waits ~43s on the rate limit, hidden behind beat 2's ~2.5 min of
  speech). Prompts, transcript (all beats, then doubts), checkpoints
  identical — verified old vs new loop under identical fakes (4 scenarios
  incl. interruption cap). A beat-2/3 LLM error now surfaces after beat 1
  was spoken (before: before any speech); the section still re-runs on
  resume. `teach_section_in_beats` kept (= list(iter_section_beats)).
  Embedding model is preloaded in a thread at main() start
  (`vector_index.preload_model_in_background`, `_get_model` has a lock),
  and plan_lecture speaks the greeting BEFORE assign_context (2026-09-28)
  so the load overlaps the greeting — no audio underruns measured.
  `_get_model` loads with `local_files_only=True` first (online fallback
  only if not downloaded): the HF Hub online check cost ~7s, once 69s.
  Measured DBMS startup: greeting starts at ~8s (was ~20s of silence),
  assign_context 0.1s after it, pause greeting → first teaching sentence
  ~3.7s (unchanged), first sentence ~27s after launch (was ~78s).
- **quiz.py** — post-lecture quiz. Gemini (`gemini-3.8-flash`) first with
  the full payload; per-minute 429 → wait+retry once; per-DAY quota
  (free tier ≈ 20 req/day/model) → no retry, falls back to Groq with a
  trimmed prompt; if both fail raises `QuizUnavailable`. classroom.py
  catches it and exits cleanly — checkpoint stays parked at the quiz
  node, so rerunning the same command resumes straight into the quiz.
  Reloads the full source via `_load_full_document` (YouTube →
  `rag.load_youtube`; extension-less files are content-sniffed).
- **teacher.py** (Phase 0 text prototype; also provides `call_llm`,
  `USE_REAL_LLM`), **teacher_natural.py** (persona prompts, `speak`,
  `RESUME_PHRASES`), **teacher_voice.py** (voice graph pieces:
  `VoiceTeachState`, sentence splitting, `MAX_INTERRUPTIONS_PER_SECTION=6`),
  **voice_io.py** (pyttsx3 TTS with barge-in via sounddevice RMS
  detection, record-until-silence, Google free STT), **tts_utils.py**
  (`fresh_engine()` — clears pyttsx3's weakref engine cache; fixes the
  Windows "TTS skips" bug; selects female voice Zira → Hazel, rate 150).
- **neural_tts.py** — teacher voice = edge-tts neural voice (`NEURAL_VOICE`,
  default `en-US-AvaNeural`, rate -5%; free, needs internet; deps
  `edge-tts` + `soundfile` for MP3 decode). Tried first by
  `teacher_natural.speak` and `voice_io.speak_interruptible` (barge-in via
  `sd.stop()`); any failure → pyttsx3/Zira fallback, and a network failure
  disables neural for the rest of the run. ~1.1–1.6s synth per request,
  so classroom calls `prefetch()` on a section's sentences (and on doubt
  answer + resume phrase) to remove per-sentence gaps.
- **persistence.py** — `classroom_progress.db`: per-session status
  (in_progress / completed). Thread id = hash of (student, topic), so the
  same student+topic resumes the same LangGraph thread; a completed one
  starts a `-retry-<ts>` thread.
- **classroom_checkpoints.db** — LangGraph SqliteSaver checkpoints (resume
  after crash/close). If a bad plan got saved, renaming/deleting this file
  forces a fresh plan. A backup `classroom_checkpoints_backup_2026-09-27.db`
  exists (holds an old broken ```json-outline session). As of 2026-09-27
  the main file doesn't exist (only the backup) — it's recreated on the
  next run.
- On resume, `main()` re-calls `set_active_index(build_index(saved_path))`
  (using the checkpoint's `document_path`) when the next node is
  `teach_and_listen` — `plan_lecture` is skipped on resume, so without this
  `search_document` had no index.
- Side scripts: `plan_demo.py` (calls `planner.plan_lecture`, which is NOT
  the graph node in classroom.py — its chunk_size/overlap defaults are
  200/40 to match classroom), `rag_demo.py`, `history.py`. The graphs
  inside teacher.py / teacher_natural.py / teacher_voice.py are old
  standalone prototypes; classroom only imports helpers from them.
- Coverage check (classroom `_source_coverage_ok`, fresh PDF lectures
  only, before the graph): pages that contributed no chunk
  (`rag.ingest.missing_pdf_pages`) → warn below 80% coverage, ask
  "Continue anyway? [y/N]" below 50% (no/EOF → exit 3).
- Topic default: file name → title-cased; YouTube → real title via oEmbed
  (`_fetch_youtube_title` in classroom.py; `rag.get_video_title` also exists).

### rag/ package
- `ingest.py` — loaders per extension; PDFs via pdfplumber with table
  detection, diagram detection, garbled-text detection, OCR fallbacks;
  structure-aware `chunk_elements`. Diagram detection (since 2026-09-27) =
  `_find_diagram_regions`: clusters of DRAWN vector shapes (rects/lines/
  curves); all words inside a region → one diagram element (vision-
  described, labels as fallback). Ignores page-spanning lines (ruled
  paper), tiny dots, table grids, underline-only clusters. Only other
  diagram trigger: `is_garbled_text`. The old words-per-line
  `_looks_like_diagram` is kept but UNUSED (it flagged bullet lists as
  diagrams and missed real flowcharts). Tested against
  `sample_test_document.pdf` + hand-built cases (bullets, ruled paper,
  underlines, table, flowchart).
  Text-ish loaders (txt/md/html/json) read via `_read_text_file`: UTF-8
  (BOM ok) → cp1252 → UTF-8 with replacement; csv retries with cp1252.
  Extension-less paths are content-sniffed (`_sniff_extension`: pdf /
  docx / pptx / xlsx / png / jpg / html, else txt) inside ingest, so
  lecture + quiz both handle them (quiz's own older sniffing still exists).
  Two-column pages (since 2026-09-27, cache `v4-columns`):
  `_find_column_gutter` finds an empty middle strip; accepted only if both
  sides are prose columns (>=25% of words, >=4 words/line, >=30% width,
  and `_vertical_fill` >= 0.8 — rejects borderless 2-col tables; measured
  text columns 0.94–1.0 vs Unit 1 p2 Period|Milestone table 0.63). Then
  `_split_columns` (per row, so full-width titles stay whole) +
  `_column_reading_order` (bands split by full-width items: full-width →
  left → right). Gutter None → exact old path; verified byte-identical
  output on Unit 1 PDF, sample PDF, diagram cases. Elements carry a temp
  `_bbox` inside `_load_pdf` only (popped before return). Test PDFs:
  scratchpad `make_columns_pdf.py` (hand-written PDF, no libs).
- `page_ocr.py` (since 2026-09-27, cache `v5-ocr`) — local OCR for PDF
  pages with NO text layer (slide screenshots/scans), e.g.
  `DBMS_U_5_ONE_SHOT.pdf` (71/72 image-only pages: 1 → 72 pages read).
  RapidOCR (`rapidocr_onnxruntime`, pip-only, ~2s/page CPU); dominant
  embedded image OCR'd at native res via pypdfium2 (skips watermark
  overlays), else page rendered at 200 dpi (no poppler/tesseract needed).
  Dropped English spaces restored by wordninja boundary + char-box gap
  ≥0.20×median width (35 hand-checked lines: 50% → 98% word accuracy).
  Lines <0.8 confidence dropped (handwriting); short lines on ≥30% of
  pages dropped (logos). Output = pdfplumber-style words → same layout
  code as text pages. Cached per image in `.ocr_cache/` (full DBMS PDF:
  141s first, 13s cached — quiz reload relies on this). OCR'd pages skip
  vision and the garbled→diagram path. Column split on OCR pages also
  allows non-prose sides when rows don't align (`_row_alignment` < 0.6;
  p16 0.0, p33 0.18, p69 0.5) — text-layer PDFs NOT affected (a
  centred-cell table measured 0.52 there). No rapidocr → old behaviour.
- Smarter vision (step 3, 2026-09-28, cache `v6-vision`): image-only
  pages that OCR can't do justice (`page_ocr._vision_reason`: nothing
  read / ≥30% lines <0.8 conf / grid table = ≥3 columns aligned within
  1% width across ≥4 rows — looser rules flagged justified prose) get ONE
  batched whole-page transcription (`vision_describe.transcribe_pages`,
  ≤4 pages/request, verbatim prompt: left block then right, tables as
  `[TABLE]` rows "Label — Header: value; …", code kept, `[Diagram: …]`),
  cached in `.vision_cache/transcriptions.json`; failures keep OCR.
  `_transcript_elements` → text/table/diagram elements. Image loop skips
  decoration (`_is_decorative_image`: content repeated on ≥3 pages & ≥30%;
  ≥90% page background behind a text layer; <10% page and touching the
  edge). Daily ingestion vision budget `VISION_DAILY_BUDGET` (default 12,
  `.vision_cache/usage.json`, Pacific day) leaves quota for the quiz.
  Also fixed two #1 diagram-detector bugs: bordered prose/one-line text
  boxes (`_is_text_box`: ≤1 line or >6 words/line) were "[Diagram]" (18
  Unit 1 paragraphs; with vision on they'd be REPLACED by summaries), and
  table border lines on the bbox edge escaped exclusion (padded 2pt; small
  text-less regions <5% page skipped). Vision calls: Unit 1 30 → 8,
  DBMS up to 146 → 2, sample 2 → 2. Vision-off output identical except
  those text boxes becoming text (Unit 1 words == original extraction).
  Real transcription not yet verified: Gemini returned 503 on 2026-09-28.
- TC-U-5-ONE-SHOT-Notes.pdf held-out test (2026-09-28, cache `v7-tc`;
  PPT export, text layer, bullets, 2 image figures) — general fixes in
  `_load_pdf`: `page.dedupe_chars()` (same-position fake bold, DBMS p1) +
  `_dedupe_overlapping_words` (same word drawn twice with an offset =
  shadow/outline titles, TC p1; ≥50% box overlap; dedupe_chars NOT
  loosened — genuine double letters sit 2.0pt apart); 
  `_drop_repeated_headers_footers` (text/table element in top/bottom 15%
  repeating on ≥30% of pages, ≤15 words — TC footer/header were 14% of
  words and the footer boxes were a fake "table" on 56 pages); private-use
  symbol-font glyphs → "•"; content images on text pages get local OCR
  offline (`_ocr_region`: Tesseract if usable, else
  `page_ocr.ocr_image_text`; kept if ≥3 words) — image loop now runs when
  vision OR OCR is available; `_is_decorative_image` also skips images
  <1.5% of the page (TC emoji/icons: 13 would have used the vision budget
  before the real figures). Results: TC boilerplate removed, figures p38
  (Maslow pyramid)/p42 (Freud) readable offline, 8/8 test questions
  retrieved at rank 1, vision calls 19 → 6; Unit 1 gained OCR text of its
  8 figures; other PDFs unchanged. Still open: TC p10 bullet hidden
  behind the footer is garbled into it; title/end-slide shapes cost ~4
  vision calls; OCR of low-res figure images is imperfect ("lack oi").
- Image files as the source (#10, 2026-09-28): `.png/.jpg/.jpeg/.webp/
  .bmp/.tif/.tiff` → `ingest._load_image_structured` (before: straight to
  Tesseract → TesseractNotFoundError here). EXIF rotation fixed
  (`ImageOps.exif_transpose`; a sideways phone photo read 97% like the
  upright image vs 13% without), RapidOCR via `page_ocr.ocr_image_words`
  (boxes scaled so the longest side ≈1000 units → same layout thresholds
  as slide pages), `_elements_from_words` (column split with
  allow_unaligned), and ONE `transcribe_pages` vision call when
  `_vision_reason` says OCR falls short (nothing read / handwriting / grid
  table); Tesseract last; clear error only if no OCR and no vision. Single
  images get no multi-page boilerplate removal (logos stay). Known: the
  image path spaced "transaction,so" where the PDF path gave
  "transaction, so" for the same slide — cause not yet found.
- Incomplete-index handling (step 5, 2026-09-28): `vision_describe`
  counts a "shortfall" whenever vision was wanted but temporarily
  unavailable (quota/503 block, daily budget used up, failed call, page
  missing from a transcription reply — NOT when there's no key at all).
  `vector_index.build_index` compares the counter before/after ingestion
  and stores `"incomplete": true` in meta.json. `_cache_is_valid` then
  rebuilds that index on a later run once `vision_available_now()` (only
  missing items call the API; OCR/vision results are cached), reuses it
  while vision is still unavailable, and retries at most once per process
  (`_attempted_this_process` — classroom calls build_index twice at start).
  Old caches without the flag count as complete. 10 fake-Gemini tests +
  identical indexes on all real PDFs with vision off.
- Code blocks vs garbled check (2026-09-28, cache `v8-code`):
  `is_garbled_text` returns False when text has ≥1 code/math operator per
  10 tokens (`_CODE_OR_MATH`: `= + < > ← ≤ ≥ ≠` or `word(`). Before, short-
  token pseudocode ("X:= X+ Y;", "T1: read(X); X := X - 50;") counted as
  garbled → on text-layer PDFs it became "[Diagram …]" (vision summary or
  "fragments not shown" placeholder) instead of the code. Chopped labels
  (": Cl; : ie; : n;") have no operators, so they're still caught, even
  with one stray "=". Tested: scratchpad `code_cases.pdf` (3 code blocks
  now text, garbled block still placeholder); all 1534 word blocks of all
  test PDFs — only the code blocks changed; ingested output (vision off)
  byte-identical on sample, diagram/columns cases, Unit 1, DBMS, TC and
  the 5 test images. Cache bumped because older indexes of code-heavy PDFs
  hold the placeholders.
- `vision_describe.py` — Gemini vision descriptions of diagrams/images.
  NOTE: spends from the SAME Gemini daily quota as the quiz (one call per
  diagram). `.vision_cache/` exists (description cache + quota circuit
  breaker `quota_state.json`). Client uses `max_retries=0`. Breaker
  duration: per-DAY error → until midnight Pacific (fixed UTC-8; the
  error's "retry in 36s" hint is ignored for daily quota); per-minute →
  Gemini's hint; 429 without hint → 60s; 503 UNAVAILABLE ("high demand")
  → 5 min block (never shortens a longer one) — a real run lost ~42s to
  three 503s on page-1 images.
- `requirements.txt` has duplicate lines (23–37 repeat 1–14) — harmless.
- `youtube_ingest.py` — transcripts via youtube-transcript-api (old+new API).
  `is_youtube_source` accepts any subdomain (www/m/music), scheme
  optional; `extract_video_id` handles watch / youtu.be / embed / shorts /
  live.
- `vector_index.py` — DEFAULT backend: sentence-transformers
  `all-MiniLM-L6-v2` + FAISS, cached in `.rag_cache/` (YouTube keyed by
  video id). Folder = readable prefix + hash (`yt-<videoid>-<hash>`,
  `<file-stem-slug>-<hash>`); the hash alone decides identity. Hash-only
  folders are from before 2026-09-27 (cache v2) and are unused leftovers. `document_index.py` — TF-IDF offline alternative.
- `retrieval.py` — neighbor expansion / low-quality filtering.
- `websearch.py` — ddgs free web search.

## Known constraints / gotchas
- Groq free tier: 8000 tokens/min per model — keep prompts small; this
  has crashed the Planner before.
- Gemini free tier: ~20 requests/day/model — vision + quiz share it.
- LLMs sometimes wrap JSON in ``` fences — always parse defensively.
- A section's 3 beat calls fire back to back (all generated before any
  speech). Measured on a real lecture: each call ~3.0–3.7k tokens (fine),
  but the 3 together ~9–11k/min + outputs + gpt-oss reasoning tokens →
  over 8000 TPM. Handled by the retry above; per-section context is 10–12
  chunks (~1500–1850 words) because neighbor expansion ~doubles k=5.
- No mic/speaker in cloud sandboxes; voice must be tested on the laptop.
- Import side effects: importing `teacher_natural` probes the TTS engine,
  `voice_io` imports sounddevice — so `import classroom` needs audio libs.
- Git repo since 2026-09-28 (first commit `bcae1f8`; this repo's
  `user.email` = kaustubhjeet2019@gmail.com, set locally). `.gitignore`
  ignores `.env`, `.venv`, caches, `*.db` session data, PDFs except
  `sample_test_document.pdf`, generated `.wav/.mp3`. `.gitattributes`
  marks pdf/images/audio binary (git misdetected the sample PDF as text;
  Windows CRLF conversion would have broken it).

## Fragile coupling — check before changing
- `ClassroomState` fields / meaning of `idx`, `outline`, `section_context`
  are stored in checkpoints; changing them can break resuming saved lectures.
- Chunk format change → bump `_CACHE_VERSION` in `rag/vector_index.py`,
  else stale `.rag_cache` entries get loaded.
- Offline stub LLMs match on exact prompt phrases: `"Return ONLY a JSON
  list"` and `"as a PhD-level teacher"` — rewording prompts changes stub output.
- Empty-document path uses `DocumentIndex()` (TF-IDF class) while normal
  path returns `VectorIndex`; works by duck typing (`is_empty()`, `chunks`)
  — keep both interfaces aligned.

## Known open issues (not fixed yet)
- If `MAX_INTERRUPTIONS_PER_SECTION` is hit, remaining beats are skipped
  but the transcript still records the full section text → quiz may ask
  about unspoken content. Why the cap exists (pre-existing, from the
  teacher_voice prototype): barge-in is loudness-only (RMS > 300); a false
  trigger (noise / speaker echo) → empty transcript → the SAME sentence
  restarts → can loop forever. But it also counts real doubts and skips
  silently. Agreed design (deferred 2026-09-28, not urgent): count only
  consecutive EMPTY interruptions on the same sentence; when hit, pause
  barge-in for the rest of the section and keep teaching (printed note) —
  real doubts never capped, nothing skipped. Needs a mic-level
  measurement on the laptop + fake-audio old-vs-new loop test first.
  Revisit if the teacher restarts a sentence by itself or skips ahead.
- Quiz reloads the full PDF through `_load_pdf`, which calls vision for
  diagrams; free if `.vision_cache` hits, but spends Gemini quota (the same
  pool the quiz needs) if the cache was cleared.
- Unused code: `explainer.build_grounded_teach_prompt`, `rag.get_video_title`,
  `rag.ingest._looks_like_diagram`.
- MiniLM 256-token truncation — MEASURED 2026-09-27, deliberately NOT
  fixed: YouTube chunks never exceed it; the AI Unit 1 PDF has 11/29
  chunks over (5% of tokens unembedded), yet 7/7 phrases taken from the
  cut-off tails were still retrieved at rank 1 (40-word chunk overlap +
  semantically uniform chunks). Max-pool windows gave the same recall but
  shift scores up ~+0.06 (would move what passes min_score 0.15/0.3);
  mean-pool lost 1/11 head queries. Revisit only if chunks get much longer
  (bigger chunk_size, long tables/vision descriptions).
- DBMS_U_5_ONE_SHOT.pdf review, still open (steps 4-5): page 1
  text-layer has doubled glyphs ("AA KK TT UU" → pdfplumber
  `dedupe_chars`); page 1's multi-line header block is still one
  "[Diagram]"; near-duplicate pages (intro slide ×4); ~"timestamp" split
  4/65 by OCR spacing. (Caching of vision-degraded indexes: fixed in
  step 5.)
- rag weak spots found 2026-09-27, still open: cached
  index never upgrades once vision quota returns (fixed by step 5);
  `filter_low_quality` not used by planner/tools — MEASURED 2026-09-28,
  deliberately NOT added: 0/107 current chunks and 0/354 old-cache chunks
  are flagged (garbled text is caught at ingestion since #1; old caches
  retired by version bumps), and it would false-flag short pseudocode
  (DBMS "X:= X+ Y; Y := X + Y;"). The related risk (the ingestion-time
  garbled check sending TEXT-LAYER code blocks to the diagram path) was
  FIXED 2026-09-28 — see "Code blocks vs garbled check" above.

## How Kaustubh wants you to work
- Make ONLY the changes needed; no unrequested refactors or rewrites.
  Don't risk breaking working code. Verification should match the change:
  a small, local edit → just compile/check that file; a large change or
  one to a core part (graph/state, planner, explainer, rag, quiz,
  persistence) → full compile + import of all modules + targeted tests.
- Explain the root cause briefly, then fix. Minimal explanation.
- Ask before deleting anything; prefer reversible changes (rename/backup).
- He is strong in ML (tabular, XGBoost), learning GenAI/agentic AI; no
  frontend background — explain frontend work if any comes up.
