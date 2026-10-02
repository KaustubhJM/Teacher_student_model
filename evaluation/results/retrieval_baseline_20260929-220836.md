# Retriever baseline — golden/dbms_u5.jsonl

- run: 20260929-220836 · judge `groq/openai/gpt-oss-120b` · retriever search_document baseline {'k': 3, 'window': 1, 'neighbor_min_score': 0.3}
- index: {'DBMS_U_5_ONE_SHOT.pdf': {'chunks': 38, 'fingerprint': 'f2c51b079de7', 'incomplete': True, 'cache_version': 'v8-code'}}
- search_document match: 20/20

| group | n | recall | sufficient | precision (sent order) | precision (score order) | P@k | 1st useful rank | chunks | words |
|---|---|---|---|---|---|---|---|---|---|
| **in-document** | 18 | 0.91 | 78% | 0.90 | 0.90 | 0.65 | 1.06 | 6.00 | 1175.28 |
| comparison | 2 | 1.00 | 100% | 0.89 | 0.89 | 0.87 | 1.00 | 8.00 | 1577.00 |
| definition | 5 | 0.80 | 80% | 0.89 | 0.89 | 0.60 | 1.00 | 5.40 | 1059.00 |
| explanation | 4 | 0.95 | 75% | 0.93 | 0.93 | 0.63 | 1.00 | 5.25 | 1029.50 |
| multi-hop | 3 | 0.85 | 33% | 0.80 | 0.80 | 0.73 | 1.33 | 7.67 | 1506.00 |
| table | 2 | 1.00 | 100% | 0.92 | 0.92 | 0.30 | 1.00 | 5.00 | 958.50 |
| worked-example | 2 | 1.00 | 100% | 1.00 | 1.00 | 0.83 | 1.00 | 5.50 | 1076.50 |

## Out-of-document questions

In-document top-1 similarity: min 0.22, median 0.66, max 0.83

- dbms_u5-019: top-1 similarity 0.19, useful chunks 0/3
- dbms_u5-020: top-1 similarity 0.55, useful chunks 3/7

## Weakest in-document questions

- **dbms_u5-004** (definition) recall 0.00, precision 1.00 — gold pages [48], retrieved pages [49, 50, 51, 56, 57, 58, 59, 60]
  - missing: The phantom phenomenon occurs when a transaction retrieves a set of rows satisfying a condition and another transaction inserts or deletes rows that also meet that condition.
  - missing: The phantom phenomenon leads to inconsistent query results because the retrieved data changes during the transaction's execution.
  - missing: Transaction T1 reads all rows with salary > 5000.
  - missing: Transaction T2 inserts a row with salary 6000.
  - missing: Transaction T1 sees an extra phantom row when transaction T1 re-runs the query.
- **dbms_u5-011** (multi-hop) recall 0.71, precision 0.65 — gold pages [21, 23, 26, 31], retrieved pages [1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 23, 24, 25, 49, 50, 51, 52, 53, 54, 55, 56]
  - missing: Acquiring all locks before execution can cause delays due to pre-locking.
  - missing: A transaction T waits for another transaction T' only if TS(T) > TS(T').
- **dbms_u5-008** (explanation) recall 0.82, precision 1.00 — gold pages [32], retrieved pages [29, 30, 31, 32, 33, 69, 70]
  - missing: If write_TS(X) > TS(T), the write is already outdated and obsolete.
  - missing: If neither read_TS(X) > TS(T) nor write_TS(X) > TS(T), the write is executed.
- **dbms_u5-010** (multi-hop) recall 0.83, precision 0.93 — gold pages [21, 43, 44, 63], retrieved pages [18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
  - missing: A transaction only reads committed data under Strict 2PL.
- **dbms_u5-005** (definition) recall 1.00, precision 0.70 — gold pages [42, 44], retrieved pages [1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 23, 24, 25, 40, 41, 42, 43, 44, 45]
