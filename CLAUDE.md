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
- **evaluation/** (2026-09-28) — RAG eval metrics, standalone (NOT wired
  into classroom). `AnswerRelevancyMetric` = DeepEval's algorithm, local
  (no deepeval dep): judge (Groq `openai/gpt-oss-120b`, temp 0, or any
  `llm=` callable) splits the answer into statements, labels each
  yes/idk/no vs the question; score = (yes+idk)/n, `strict_mode` → 0
  unless all relevant; empty answer → 0. Reason built locally (2 judge
  calls/case). Tutor filler ("Good question, Kaustubh!") is forced to
  "idk" — without that rule the judge scored a perfect answer 0.75.
  CLI: `python -m evaluation cases.json [--threshold] [--strict]`
  (JSON list/JSONL of {input, actual_output}; `sample_cases.json` →
  1.00 / 0.33 / 0.00). Same judge model as the explainer (self-judging
  bias possible; pass `model=`).
- **Retriever eval** (2026-09-29, deepeval 4.2.6 installed; it pins
  click<8.4 → hf-hub's click>=8.4.2 conflict, only affects the `hf` CLI,
  embeddings verified OK). `golden/dbms_u5.jsonl` = 20 Q + ideal-answer
  pairs written from the PDF slide images (18 in-doc, 2 out-of-doc;
  schema/lifecycle in `golden/README.md`, room for LLM drafting + human
  review). `evaluation/judge.py` = `GroqJudge(DeepEvalBaseLLM)`
  (gpt-oss-120b — llama-3.3-70b is NOT on this key; key limits 1000
  req/day, 8000 TPM, **200k tokens/DAY (TPD), shared with the explainer**;
  ~1300 tokens/judge call incl. reasoning). A baseline run ≈150k tokens;
  baseline + sweep exhausted the TPD on 2026-09-29 (sweep stopped at
  5/20, cached) — after a big eval run, lectures fail with 429 until the
  usage rolls off (~24h). Run evals when no lecture is planned.
  `evaluation/retrieval_metrics.py` = custom DeepEval metrics
  `ContextualClaimRecallMetric` (claims of ideal answer found in chunks)
  + `RankedContextualPrecisionMetric` (DeepEval weighted cumulative
  precision + P@k), one cached judge call per (question, chunk) in
  `evaluation/.judge_cache/` (key includes model + PROMPT_VERSION + hashes)
  → re-scoring other k/window/order is free. `evaluation/retrieval_eval.py`
  replicates `search_document` (asserts equality with the real tool) and
  scores precision in sent (document) order AND score order; results in
  `evaluation/results/`. Judge smoke test 6/6 (it is lenient on a
  same-condition rule from another protocol). DeepEval writes `.deepeval/`
  in cwd (git-ignored). Run evals with `GOOGLE_API_KEY=` (empty) so an
  INCOMPLETE index isn't rebuilt by vision mid-eval (fingerprint check).
  Results (18 in-doc Qs): current search_document recall 0.91, sufficient
  78%, P@k 0.65, ranking precision 0.90 best-first but 0.68 in the
  document order actually sent, ~1175 words. Sweep: knee at pure top-4
  (recall 0.92, P@k 0.76, 780 words, best F1); window 2 adds nothing;
  direct-hit min 0.3+ loses recall. Neighbours win continuation/multi-hop
  Qs (007, 010, 011), top-4 wins 004 (phantom chunk ranks 4th; as a
  neighbour it inherits 0.9×0.22 < 0.3 and is dropped). k=4 + window 1:
  recall 0.96 / P@k 0.64 / 1426 words — but the whole gain is 004 alone
  (an ingestion symptom), so NOT adopted; fix slide-48 columns first.
  **Applied 2026-09-29 (approved): `tools.search_document` returns its
  chunks best-first** (one `sorted()` after expand_with_neighbors; same
  chunks). DBMS: ranking precision 0.68 → 0.90, useful chunk first 17/18
  (was 5/18); recall/P@k unchanged (0.91/0.65); eval harness updated,
  search_document match 20/20. Planner/lecture path deliberately still
  page order (teaching flow, not measured). "Passages best-first" (keep
  adjacent runs together) scored only 0.77 — the passage starts with the
  pre-hit neighbour, usually not useful. Effect on spoken answers not yet
  measured (generation eval).
  P@k (share of sent chunks judged useful) is capped by the embeddings:
  neighbour cut-offs on inherited OR own cosine only reach 0.68–0.69
  (useful vs useless neighbours overlap: median own cosine 0.55 vs 0.44).
  Cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU,
  ~0.6–0.9 s per 10 chunks, ~90 MB in HF cache, no new dep) separates them
  (useful median +0.54 vs −5.40). DBMS: "rerank top-10, keep ≥ −2.1, max 8"
  recall 0.96 / P@k 0.76 / 1054 words, BCNF gets no chunks; recall gain is
  again only 004. Threshold picked on DBMS → both reranker settings are
  in SWEEP for the held-out TC check. NOT in tools.py until TC confirms it.
  TC full sweep pool = 12 chunks/Q (43% of the 28-chunk index, ~650 calls
  ≈ 4 days) → `--sweep --core` skips the "wide" settings (top-6/8/10,
  window 2, k4 w1; settled on DBMS): 7.2 chunks/Q (~410 calls ≈ 2.5
  days). 2026-09-30: TC 001–012 judged with the full pool, rest --core.
  TC complete 2026-10-02 (42 in-doc): current recall 0.99 / sufficient
  95% / multi-hop 0.97 / P@k 0.50 / 1145 words. Reranker at −2.1 FAILED
  the held-out check (recall 0.91, multi-hop 0.73) — it drops pieces of
  spread-out answers. Filter on today's chunks at rr ≥ −8: NO recall loss
  on either PDF (DBMS 0.91/78%/0.85, TC 0.99/95%/0.97), P@k DBMS 0.65→0.78,
  TC 0.50→0.75, words −16/−30%, off-doc chunks 10→3 / 37→7. "Always keep
  top-2/3" variants are worse (keep junk for off-doc Qs). −8 chosen after
  seeing both sets (no held-out left).
  **Applied 2026-10-02 (approved): reranker filter in
  `tools.search_document`** — after k=3 + neighbours, `rag/reranker.py`
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `RERANK_MIN_SCORE = -8.0`)
  scores the chunks; keeps rr ≥ −8 ordered by rr; none left → "No
  relevant content found in the document for this query." Reranker
  unavailable (not downloaded + offline, load/predict error) → today's
  behaviour, verified identical 20/20. Re-verified from cache: DBMS 0.91 /
  78% / P@k 0.78 / ranking 0.93; TC 0.99 / 95% / P@k 0.75 / ranking 0.96;
  tool == harness 20/20 + 50/50. Cost: +~0.23 s per doubt (max 0.45 s),
  load 6.2 s from disk. `classroom.main()` calls
  `preload_reranker_in_background()`, which waits for the embedding model
  first (no CPU contention in the tuned startup). Real lecture startup NOT
  re-measured yet (needs Groq + audio). Planner/lecture path unchanged.
  Second golden set `golden/tc_u5.jsonl` (2026-09-29): 50 Qs on
  TC-U-5 notes (42 in-doc incl. 4 figure Qs on the Maslow pyramid / Freud
  image, 8 out-of-doc of which 6 near-misses); every content slide 5–57
  covered. TC index = only 28 chunks (58 slides), so k=3+neighbours sends
  ~20% of the PDF per question; built with vision → Gemini 503 → 6
  figure/title pages OCR'd, INCOMPLETE, fingerprint `af70e47cf8fe`.
  Baseline ≈350 judge calls ≈2 days of TPD — run in batches.
- **Generation eval** (2026-10-03) `evaluation/generation_eval.py`: calls
  the REAL `explainer.answer_doubt` per golden question (state =
  Kaustubh + topic from file name, section "Revision of the whole unit"),
  records tool calls by temporarily swapping `explainer.search_document /
  search_web` for recorders (answer_doubt looks tools up by name at call
  time; schemas are bound first), queries, latency, teacher tokens
  (`get_usage_metadata_callback`). Scores: DeepEval FaithfulnessMetric
  (default = only contradictions count; "groundedness" = strict, from the
  same verdicts), local AnswerRelevancyMetric, correctness = ideal-answer
  claims stated/contradicted (claims read from the retrieval gpt-oss
  cache). Answers + scores cached in `evaluation/.judge_cache/
  generation.json`. Judge = **qwen/qwen3.8-27b** (own quota, not the
  teacher's family; `reasoning_format="hidden"`), smoke test 4/4, agreed
  exactly with gpt-oss on 4 real answers; quirk: qwen marks WRONG but
  on-topic statements irrelevant (relevancy 0 on a contradicting answer).
  Findings on 4 Qs: teacher answers standard textbook doubts (2PL) with NO
  tool (prompt allows general knowledge) — correct anyway; BCNF: reranker
  "No relevant content" → teacher answered correctly from knowledge;
  notes-specific Qs use search_document (rewrites/repeats queries);
  groundedness caught an embellishment ("large majority" vs 40%);
  correctness caught an omission (Amit Agarwal). Eval latency is inflated
  by Groq TPM waits (judge + teacher share minute limits when same model).
  Cost: teacher ~0.8–4.6k tokens/Q (gpt-oss TPD), qwen judge ~1.4–6.2k/Q.
  qwen3.8-27b limits: 200k tokens/DAY too, AND 1000 OUTPUT tokens/minute
  — a single request asking for >1000 output tokens fails ("Request too
  large ... OTPM"; hit on DBMS 011 / TC 016, long contexts → DeepEval
  truths extraction). Full run 2026-10-04 stopped at TC 028 (qwen TPD).
  Partial results, 45/70 scored (43 in-doc): faithfulness 0.97,
  groundedness 0.59, relevancy 1.00, correctness 0.80; only 25/43 answers
  used document context, 17 used NO tool. Correctness with doc context
  0.84–0.85 vs without 0.70–0.75 (worst: MBTI 0.20, 2PL concurrency 0.20,
  perfectionist 0.29 — all no tool). Cause: doubt prompt says general-
  knowledge questions don't need tools → teacher answers textbook topics
  from memory, not from the student's notes. Real errors vs notes
  (spot-checked, judge fair): 008 Thomas's rule "preserves serializability"
  (notes: does NOT enforce conflict serializability); 010 calls strict 2PL
  "rigorous" + "MV2PL works with versions instead of locks"; 015 wrong
  read_TS. Candidate fix (NOT applied, core prompt change → plan + yes):
  "search the notes first for course questions"; measure with this eval.
  FINAL 2026-10-05, all 70 scored, 0 errors (results
  `generation_20261005-004518.json` DBMS + `-004420.json` TC): 60 in-doc
  faithfulness 0.97 / groundedness 0.60 / relevancy 0.98 / correctness
  0.77 (22/60 fully correct, 9 < 0.5); doc context 39/60, no tool 20/60;
  correctness with doc ctx 0.79 vs without 0.72. By type: figure 0.44
  (worst — 037 answered the TEXT slide's safety needs, not the pyramid
  figure's; TC figures were OCR'd because Gemini vision 503'd), explanation
  0.69, multi-hop 0.75, definition 0.80, comparison 0.84, list 0.87.
  Off-doc (10): correctness 0.92, only 1/10 got doc context, web 2/10 —
  not misled by unrelated slides. "Answer contradicts its own context"
  (9 answers) is NOISY with qwen: spot-check 035 real (mixed figure with
  8-stage model), 028 borderline paraphrase, 030 judge false positive.
  OTPM fallback (truths_extraction_limit=25) never triggered on rerun —
  earlier failures were per-minute budget, not single-request size.
  Teacher ~1.7k tokens/Q average.
  **Applied 2026-10-05 (approved): doubt prompt change** in
  `explainer._doubt_system_prompt` — the "general knowledge questions don't
  need tools" paragraph replaced by: questions about the subject → ALWAYS
  call search_document first, base the answer on the notes' definitions/
  lists/examples, follow the notes where they differ from own knowledge,
  own explanation only on top, search_web only for what the notes lack,
  skip tools only for comprehension-check answers/comments. Tool
  descriptions, history replay, 2–5 sentence style unchanged. Expected
  cost: +1 tool round (~+1–2 s, ~+2k tokens) on doubts previously answered
  from memory. Success criteria agreed: in-doc correctness 0.77 → ≥0.82,
  doc-context rate 65% → ≥90%, faithfulness ~0.97 kept, off-doc
  correctness ≥0.90 without leaning on unrelated slides. Measuring run
  started 2026-10-05 (answer cache key includes the prompt hash, so new
  answers are generated; old results kept for comparison). Undo = git.
  Partial (2026-10-05, 14 DBMS in-doc Qs, stopped at qwen TPD after 015):
  correctness 0.79 → 0.92, doc context 7/14 → 14/14, fully correct 7 → 11,
  groundedness 0.58 → 0.73, faithfulness 0.97 → 0.94 (now measured on all
  14, before only on the 7 with context), relevancy 1.00 → 0.96; no
  question got worse; 008 (Thomas's rule) still partly wrong WITH the
  notes. Teacher ~1.8–4.6k tokens/doubt. qwen once returned 20 relevancy
  verdicts for 19 statements (011) → relevancy now recorded as None for
  that question instead of failing it. Remaining: DBMS 011, 016–020, all
  TC (answers for DBMS 016–018 + TC 001–003 already generated).
  Partial 2 (2026-10-06, stopped at qwen TPD after TC 032): 48 in-doc Qs
  with both prompts — correctness 0.78 → 0.90, used notes 30/48 → 48/48,
  fully correct 19 → 29, faithfulness 0.96 → 0.95, groundedness 0.64 →
  0.68, relevancy 0.99 → 0.98; 21 better / 26 same / 1 worse (TC 015
  0.88 → 0.75). The 3-tool-round crash hit TWICE in this batch: DBMS 020
  AND TC 017 (plain in-doc "Vivekananda's golden rules") — ~5% of doubts.
  FINAL 2026-10-06 (68/70 with both prompts; DBMS 020 + TC 047 crashed,
  TC 017 passed on retry; new results `generation_20261006-033235.json`
  DBMS, `-231616.json` TC, `-232011.json` DBMS 011 rescored — its OTPM
  failure was transient): 60 in-doc — correctness 0.77 → 0.87, notes used
  39/60 → 60/60, fully correct 22 → 34, < 0.5 9 → 2, relevancy 0.98 =,
  groundedness (same 40) 0.61 → 0.67, faithfulness (same 40) 0.965 →
  0.939; 24 better / 33 same / 3 worse (TC 015, 035, 038). The 20 Qs that
  used no tool before: 0.71 → 0.90. By type: definition 0.80 → 0.95,
  explanation 0.69 → 0.83, multi-hop 0.79 → 0.90, figure 0.44 → 0.54
  (037 still 0 — OCR'd figure). Off-doc (8): correctness 0.90 → 0.89
  (two small omissions in web answers), doc context 1/8, web 8/8 — not
  misled by slides. Contradiction flags 16/60 in-doc: some real (008
  Thomas's rule "for MVCC", 001 shared + exclusive at once), some judge
  noise (038). Cost: teacher tokens/doubt in-doc 1.9k → 2.4k, off-doc
  1.1k → 3.2k; off-doc median latency 1.3 s → 7.8 s (notes then web).
  Criteria: correctness ✓, doc-context ✓, faithfulness ~kept (−0.03),
  off-doc 0.89 (≈0.90, within noise).
- **Lecture eval** (2026-10-07, MEASURE ONLY — Kaustubh likes the current
  teaching, esp. the deep_dive depth; nothing in planner/explainer/
  classroom changed) `evaluation/lecture_eval.py` + `lecture_metrics.py`:
  generates one real lecture per golden file via the real
  `build_outline_from_document` → `assign_context` →
  `teach_section_in_beats` (no audio; state as classroom builds it; a
  recorder wraps `planner.expand_with_neighbors` to label direct hits vs
  neighbours), saved after every section to `evaluation/results/
  lecture_<golden>_<ts>.json` (+ .md); `--lecture FILE` continues/re-scores,
  `--sections N` = dry run. Metrics (judge qwen3.8-27b, cache
  `.judge_cache/lecture.json`): section context precision (1 batched call/
  section, direct vs neighbour split), faithfulness + accuracy (sentence
  labels notes/own/contradicts/wrong/none, ONE CALL PER BEAT — a whole-
  section call missed "2PL avoids deadlocks" that the per-beat call caught;
  "wrong" = error on something the notes don't cover, judged from qwen's
  own knowledge), teaching depth (DeepEval GEval, fixed steps), coverage
  of golden claims (per section × batches of 10 Qs) + free miss diagnosis
  from the retrieval cache (in context not taught / not retrieved by the
  planner / no known chunk), free structure stats (minutes at 145 wpm,
  comprehension check, repeated sentences cos ≥ 0.9, unused chunks).
  Pipeline hash stored so later runs know if the pipeline changed. Tests:
  7 fake-judge unit tests + qwen smoke test (labels/coverage/precision/
  depth all correct). Cost per lecture: teacher ~12k tokens/section
  (~73k DBMS), judge ~22k/section (≈ 1 qwen day per lecture).
  DBMS RESULT 2026-10-07 (`lecture_dbms_u5_20261007-001930.json/.md`,
  scoring resumed 19:41 → done 19:51): outline Locking fundamentals |
  2PL+variants | Timestamp ordering | MVCC | Phantom | Recovery+2PC;
  24.7 min speech, 6/6 comprehension checks, 0 repeated sentences, 38/38
  chunks used (14 shared). Depth 0.92 (0.8–1.0), faithfulness 0.99,
  accuracy 0.98 — 2 real errors in ~150 sentences ("2PL grow-then-shrink
  avoids deadlocks"; 2PC "servers roll back if COMMIT missing after a
  crash" — prepared participants must wait). From notes 63%. Section
  context precision 0.53 (direct 0.62 / neighbours 0.43). MVCC section
  0.08 (direct 0/5): ROOT CAUSE = assign_context cross-section dedup +
  min_score 0.15 — MVCC slides (p37–44) were claimed earlier by Timestamp
  (MVTO) and 2PL (MV2PL) sections, so MVCC got 5 "fresh" unrelated chunks
  (optimistic CC, recovery, 2PC) — reuse only happens if < k fresh chunks
  score ≥ 0.15, which almost never happens. Teacher still taught MVCC
  well from own knowledge (accuracy 1.00, depth 0.90, notes 44%) but not
  the notes' MVCC worked example. Coverage of golden claims 0.51 (3/18 Qs
  fully taught, 5 not at all: 003 granularity, 007 cost of 2PL, 013 lock
  compatibility table, 014 schedule values, 015 MVCC worked example); by
  type: definition 0.71, explanation 0.58, multi-hop 0.54, comparison
  0.40, worked-example 0.43, table 0.0. Missed claims (58): 44 "right
  section had it, not taught" (depth over breadth: ~600 words/section,
  the teacher makes its own examples instead of the notes' tables/worked
  examples), 12 "only in an off-topic section" (planner misplaced: MVCC
  example, compatibility table, granularity, conservative 2PL), 2 no
  known chunk. Nothing changed — measure only.
  TC RESULT 2026-10-08 (`lecture_tc_u5_20261007-202831.json/.md`; first
  run killed by accident during planning, restarted; scoring finished
  00:53): outline only 5 sections — Content creation intro | Content
  formats | Social-media ethics | Personality theories | Practical
  strategies for engaging content; 18.0 min, 5/5 checks, 0 repeats, 28/28
  chunks used. Faithfulness 1.00 / accuracy 1.00 (0 flags), depth 0.82
  (formats section 0.30: deep_dive spent all 233 words on ONE invented
  webinar example, beat 3 called it "a template for the other formats" —
  podcasts/blogs never explained), precision 0.67 (direct 0.76 / nbr
  0.59), from notes only 22% (section 5: 0% — thin bullet notes → teacher
  talks from own knowledge). COVERAGE 0.275, 7/42 Qs fully taught, 25 NOT
  AT ALL — incl. all 4 figure Qs, Vivekananda (017/027/028), Freud id/ego/
  superego (016/029/036/042), Maslow (024/034–038), Jung/MBTI (032/039),
  personality types (014/030/040), podcast (003/021). By type: definition
  0.82, list 0.17, explanation 0.18, comparison 0.08, multi-hop 0.04,
  figure 0.0. Missed claims: 200 right section had it not taught, 34 only
  in an off-topic section, 4 no known chunk. Causes (measured): (1) the
  OUTLINE — personality (slides ~29–57, half the unit: Maslow, Freud,
  Vivekananda, Jung/MBTI, types) got ONE section (12 chunks, precision
  1.00, 500 words), while content creation got two (sections 1 and 5 both
  use pages 1–26); (2) fixed ~500–600 words per section (3 beats) — one
  section can't teach four theories; (3) thin notes → own-knowledge
  teaching (accurate, but not the notes' exam-specific lists/figures).
  DBMS vs TC: coverage 0.51 vs 0.275, depth 0.92 vs 0.82, from notes 63%
  vs 22%, accuracy 0.98 vs 1.00, minutes 24.7 vs 18.0.
  Groq TPD is a token BUCKET refilling continuously at ~200k/24h ≈ 139
  tokens/min (≈ 8.3k/hour), not a daily reset: "Used 199926, Requested
  2663 → try again in 18m38s" (2589 over / 139 per min). So after an
  exhausted day, ~8 h of quiet gives back ~67k tokens.
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

### web/ (website frontend, started 2026-10-10)
React 19 + Vite 8 (plain JSX, plain CSS, react-router-dom), in `web/`.
Talks to the `server/` backend when it runs; otherwise PREVIEW mode on mock
data (`src/data/mockData.js`, every class `preview: true`, yellow
"Preview — not generated from your source" banner; added 2026-10-10 after
a preview "lecture" of a YouTube video was mistaken for a real one). Design = NotebookLM/Gemini Notebook dark look,
"classes" instead of notebooks; class page = Sources (left) | live lecture
captions + doubts (centre) | animated teacher "Ms. Ava" (right; SVG, mouth
driven by word pulses via `mouthRef`, later by audio loudness `level`).
`src/lib/useLecture.js` mirrors `teach_and_listen` (sentence loop, doubt →
answer → resume phrase → restart interrupted sentence). Preview voice =
browser speechSynthesis (Edge has Ava), voice doubts = webkitSpeechRecognition.
registry.npmjs.org is BLOCKED on this network (ECONNRESET) → `web/.npmrc`
uses registry.yarnpkg.com. Run: `cd web && npm run dev` (port 5173);
`vite.config.js` proxies `/api` → 127.0.0.1:8000.
Frontend data layer: `src/data/api.js` (fetch calls; `backendOnline()`
checks `/api/health` once per page load), `src/data/classStore.js` (async;
API or preview). `useLecture` (live): fetches a section's beats with a
long-poll when the section starts (also during the greeting for section
1), speaks beat 1 while beats 2–3 are still generated ("preparing" phase
while waiting), reports `sections/{n}/done`, doubts → real answer_doubt.
ClassPage polls while the class is `planning` (real plan steps), shows
`failed` with the error. QuizView fetches/caches the real quiz, saves the
score.

### server/ (website backend, 2026-10-10, approved plan)
FastAPI, text-only (the browser does voice). Run from the project root:
`uvicorn server.app:app --port 8000` (app.py chdirs to the root so
`.rag_cache/.ocr_cache/.vision_cache` are shared with the CLI). Reuses the
SAME functions as classroom.py, unchanged: plan = `build_index` →
`build_outline_from_document` → `assign_context` (background thread,
`planStep` 0–3); teach = `explainer.iter_section_beats` in a background
job per (class, section), served by `GET …/sections/{n}/beats?have=k`
(waits ≤25 s for a new beat; sentences split with the CLI's rule;
complete sections cached in the record → replay/resume costs no tokens);
`POST …/sections/{n}/done` writes the transcript exactly like
teach_and_listen (all beats, then that section's doubts; idempotent);
doubt = `answer_doubt` with the class's index set active (`_doubt_lock`
— tools has ONE global active index → single user, one doubt at a time)
and the class's doubt_history; quiz = `generate_quiz` → website shape
{q, options, answer index, why} (`_answer_index`: exact text / letter /
prefix; unmatched questions dropped), generated once and cached;
QuizUnavailable → 503. Records: `server/data/classes.db` (one JSON doc per
class; `_`-prefixed fields never sent to the browser) + uploads in
`server/data/uploads/<class id>/` (git-ignored; pasted text saved as
.txt). Env: `CLASSROOM_STUDENT_NAME`, `CLASSROOM_WEB_DB_PATH`,
`CLASSROOM_WEB_UPLOAD_DIR`. Classes: planning → new → in_progress →
completed | failed; restart keeps plan, beats and quiz.
Resume mid-section (2026-10-10, after "leaving a class resets it"):
record `position` {s, i} = next sentence to speak, saved by the browser
after the greeting and after EVERY sentence (`POST …/progress`, forward-
only so late requests can't move it back; first save turns new →
in_progress). `GET …/history` rebuilds the captions (spoken sentences up
to `position` + each doubt where it was asked, from `_doubt_log` {s, i,
question, answer}); useLecture loads it on open and starts at
`cls.position` with "Welcome back… continue with <section>". Before this
only `sections/{n}/done` saved progress, so leaving mid-section 1 showed
"Not started" and replayed from the greeting. Older records without
`position` fall back to {sectionsDone, 0}. Tests: scratchpad
`test_resume.py` (8 checks) + `test_server.py` re-run. Voice doubts:
`speech.listenOnce` now keeps the recognition error code and the page
explains it (`recognitionErrorMessage`) — Brave exposes
webkitSpeechRecognition but blocks Google's service → 'network' error,
mic lights up and nothing is heard; use Chrome/Edge.
End-of-section comprehension check (2026-10-10, website only): each
section's "connection" beat ends with ONE question; before, the site
moved straight to the next section. Now useLecture finds it
(`findCheckQuestion`: last sentence ending in "?" among the section's last
3) and enters phase `awaiting` (LecturePanel "Your turn" card: answer by
typing / mic, or Skip). Any reply (incl. "I don't know") →
`POST …/check` → `answer_doubt` with `engine._check_answer_input`: the
EXACT question + "say if it's right; if wrong / partial / don't know /
unrelated, explain the correct answer" + the reply. Skip → "No problem —
let's move on." Then section done → next section. Resume while awaiting
re-asks. Doubt memory stores `(my answer to your check "…") reply`.
explainer.py untouched. Typing just "skip"/"next"/"move on" = Skip.
Test: scratchpad `test_check.py` (8 checks, fake answer_doubt).

### CLI end-of-section comprehension check (2026-10-10, approved plan)
Shared module `comprehension_check.py` (CLI + server/engine.py):
`find_check_question` (last "?" sentence among the last beat's last 3 —
measured 11/11 sections of the saved DBMS + TC lectures),
`check_answer_input` (exact question + "say if right; explain if wrong /
partial / don't know / unrelated" + reply), `remembered_check_reply`,
`is_skip_reply` (WHOLE-reply match only: skip / skip it / next / next
question / pass / move on / let's move on …), `SKIP_LINE`.
`classroom.teach_and_listen`, after the beats only: if the LAST beat was
fully spoken (interruption cap not hit) and ends with a question →
prints "(Your turn — answer out loud, say 'skip', or stay quiet…)" →
`teacher_voice._listen_for_answer()` = `record_until_silence(
start_timeout=CHECK_ANSWER_WAIT_SECONDS=8)` (waits ≤8 s for speech to
START, then records until a 0.6 s pause, ≤15 s; empty array → '' with no
STT call) → skip words → `SKIP_LINE`, no LLM call, nothing recorded;
any other reply → `_handle_doubt(..., check_question=q)` (answer_doubt gets
check_answer_input; history + transcript store the remembered form);
silence → move on. Else (no question / cap hit): the ORIGINAL
`_listen_briefly` path. BUG FIXED with it: `_listen_briefly` used
`record_until_silence(max_seconds=4)` = 4 s TOTAL incl. thinking, so an
answer starting at 2 s was cut at 4 s (the comment claimed otherwise;
kept as is for non-question sections). `voice_io.record_until_silence`
got the optional `start_timeout` (None = byte-identical old behaviour,
verified). No ClassroomState fields added → checkpoints/resume safe.
Cost: silent student waits 8 s instead of 4 s per section (~+25 s per
6-section lecture). Tests (scratchpad): `test_cli_check.py` (33: OLD vs
NEW loop identical for silence / no-question section / mid-section doubts
/ interruption cap; "I don't know" → question+instruction; 5 skip phrases;
3 answers containing skip words NOT skipped; fake-mic default path
identical in 8 cases, late answer 11.6 s fully recorded vs 4.0 s before)
+ `test_architecture.py` (compile all, import all 37 modules, real
`classroom.main()` offline: plan → sections → simulated crash → resume
from checkpoint → quiz → completed; scratch DBs). Backups of the changed
files: scratchpad `before_check/`. Confirmed working by Kaustubh with a
real mic (2026-10-10). Default title =
CLI rule (file name title-cased; YouTube → real title via
`rag.get_video_title` while planning). Upload cap 100 MB. No CLI
coverage prompt — the source detail says "x/y pages read" instead.
Deps added: fastapi, uvicorn[standard], python-multipart. Tests:
scratchpad `test_server.py` (offline stubs, 29 checks: validation,
planning for PDF/topic/text, beats streaming + cache, doubt → history +
transcript, section done idempotent, completion, quiz shape/cache/score,
restart, delete) + curl through the Vite proxy. Known limits: a beat job
that errors mid-section regenerates from beat 1 on retry (later beats may
not match an already-spoken beat 1); no auth; one lecture at a time;
browser voice ≠ edge-tts outside Edge. NOT yet run with real Groq.

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
- `reranker.py` (2026-10-02) — optional cross-encoder relevance scores for
  the doubt tool only (see "Retriever eval"); lock + local_files_only first,
  like vector_index; never raises (returns None → caller keeps all chunks).
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
- **3-tool-round doubts (found 2026-10-06; crash FIXED, answer quality
  pending):** `explainer.answer_doubt` allows MAX_TOOL_ROUNDS=3; after
  that it calls `_get_explainer_llm()` (no tools bound), and gpt-oss still
  emits a tool call → Groq 400 "Tool choice is none, but model called a
  tool". Hit 3/4 doubts that reached round 3 in the eval (DBMS 020, TC
  047, TC 017 once). Applied 2026-10-06 (agreed): (1) any error in
  answer_doubt → spoken `DOUBT_FALLBACK` apology, lecture continues
  (fake-LLM tests 4/4; compile + import all OK); generation_eval reports a
  fallback as an error (not cached/scored). (2) a SystemMessage nudge
  ("you've used all your searches — answer now…") before the final call
  — measured USELESS: forced final call (MAX_TOOL_ROUNDS=1 in-process,
  TC 047) still 400'd 0/3 → student hears the apology. Tested (scratch,
  not applied): final call with tool-call AIMessages + ToolMessages
  removed and the results given as one plain-text SystemMessage →
  answered 6/6 (TC 047 + DBMS 020 ×3). Proposed; Kaustubh chose to KEEP
  AS IS for now (2026-10-06) — revisit if the apology is heard in real
  lectures; then rerun DBMS 020 + TC 047 in the eval (scratch test script:
  `real_final_plain.py`, monkeypatches only).
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
