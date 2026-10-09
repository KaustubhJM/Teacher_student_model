# golden/ — golden datasets for evaluating the RAG system

One JSONL file per source document (one JSON object per line).
Currently:
- `dbms_u5.jsonl` — 20 rows, `DBMS_U_5_ONE_SHOT.pdf` (scanned slides, OCR path)
- `tc_u5.jsonl` — 50 rows, `TC-U-5-ONE-SHOT-Notes.pdf` (text-layer slides + 2
  figures); every content slide (5–57) is covered by at least one question

Used by `python -m evaluation.retrieval_eval golden/<file>.jsonl`.

## Row schema

| field | meaning |
|---|---|
| `id` | stable, never reused (`<set>-NNN`) |
| `source` | document the question is about (path from project root, or a YouTube URL) |
| `question` | what the student asks |
| `ideal_answer` | the answer the retriever should be able to support (rules below) |
| `in_document` | `false` = the document does NOT contain the answer (negative test) |
| `source_pages` | pages holding the answer (for humans/debugging; not used in scoring) |
| `type` | `definition` · `list` · `explanation` · `multi-hop` · `table` · `figure` (answer only in an image) · `worked-example` · `comparison` · `out-of-document` |
| `difficulty` | `easy` · `medium` · `hard` |
| `origin` | `claude-drafted` (now) · `llm-drafted` · `human-written` |
| `review_status` | `unreviewed` · `approved` · `edited` · `rejected` (rejected rows are skipped) |
| `reviewer` | who reviewed it, or `null` |
| `version` | bump whenever `question` or `ideal_answer` changes |
| `notes` | free text |

## Rules for ideal answers
- In-document rows: written **from the document pages themselves**, not from
  the ingested/OCR'd text. That way ingestion mistakes show up as lost recall.
- 2–4 sentences, **only what is needed to answer** — every claim essential.
  The retrieved context counts as "sufficient" only when all claims are found.
- Out-of-document rows: general-knowledge answer, `source_pages: []`.

## Lifecycle (room for later)
1. Draft: `origin` = `claude-drafted` / `llm-drafted`, `review_status` = `unreviewed`.
2. Human review: set `review_status` to `approved`, `edited` (fixed the
   answer; bump `version`) or `rejected`, and fill `reviewer`.
3. Judge results are cached by a hash of the question + ideal answer, so an
   edited row is re-judged automatically on the next run.
