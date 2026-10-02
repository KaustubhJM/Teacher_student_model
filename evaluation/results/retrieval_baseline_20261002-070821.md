# Retriever baseline — golden/dbms_u5.jsonl

- run: 20261002-070821 · judge `groq/openai/gpt-oss-120b` · retriever search_document baseline {'k': 3, 'window': 1, 'neighbor_min_score': 0.3}
- index: {'DBMS_U_5_ONE_SHOT.pdf': {'chunks': 38, 'fingerprint': 'f2c51b079de7', 'incomplete': True, 'cache_version': 'v8-code'}}
- search_document match: 20/20

| group | n | recall | sufficient | precision (sent order) | precision (score order) | P@k | 1st useful rank | chunks | words |
|---|---|---|---|---|---|---|---|---|---|
| **in-document** | 18 | 0.91 | 78% | 0.93 | 0.92 | 0.78 | 1.06 | 5.06 | 989.89 |
| comparison | 2 | 1.00 | 100% | 0.97 | 0.94 | 0.93 | 1.00 | 7.00 | 1379.00 |
| definition | 5 | 0.80 | 80% | 0.99 | 0.93 | 0.87 | 1.00 | 4.00 | 784.80 |
| explanation | 4 | 0.95 | 75% | 0.94 | 0.93 | 0.80 | 1.00 | 4.75 | 933.75 |
| multi-hop | 3 | 0.85 | 33% | 0.88 | 0.82 | 0.77 | 1.00 | 5.67 | 1110.33 |
| table | 2 | 1.00 | 100% | 0.67 | 0.92 | 0.30 | 1.50 | 5.00 | 958.50 |
| worked-example | 2 | 1.00 | 100% | 1.00 | 1.00 | 0.83 | 1.00 | 5.50 | 1076.50 |

## Out-of-document questions

In-document top-1 similarity: min 0.22, median 0.66, max 0.83

- dbms_u5-019: top-1 similarity 0.19, useful chunks 0/0
- dbms_u5-020: top-1 similarity 0.55, useful chunks 2/3

## Weakest in-document questions

- **dbms_u5-004** (definition) recall 0.00, precision 1.00 — gold pages [48], retrieved pages [49, 50, 51]
  - missing: The phantom phenomenon occurs when a transaction retrieves a set of rows satisfying a condition and another transaction inserts or deletes rows that also meet that condition.
  - missing: The phantom phenomenon leads to inconsistent query results because the retrieved data changes during the transaction's execution.
  - missing: Transaction T1 reads all rows with salary > 5000.
  - missing: Transaction T2 inserts a row with salary 6000.
  - missing: Transaction T1 sees an extra phantom row when transaction T1 re-runs the query.
- **dbms_u5-011** (multi-hop) recall 0.71, precision 0.80 — gold pages [21, 23, 26, 31], retrieved pages [1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 50, 51, 52, 53, 54, 55, 56]
  - missing: Acquiring all locks before execution can cause delays due to pre-locking.
  - missing: A transaction T waits for another transaction T' only if TS(T) > TS(T').
- **dbms_u5-008** (explanation) recall 0.82, precision 1.00 — gold pages [32], retrieved pages [31, 32, 33]
  - missing: If write_TS(X) > TS(T), the write is already outdated and obsolete.
  - missing: If neither read_TS(X) > TS(T) nor write_TS(X) > TS(T), the write is executed.
- **dbms_u5-010** (multi-hop) recall 0.83, precision 0.93 — gold pages [21, 43, 44, 63], retrieved pages [18, 19, 20, 21, 22, 23, 33, 34, 35, 36, 37, 42, 43, 44, 45]
  - missing: A transaction only reads committed data under Strict 2PL.
- **dbms_u5-013** (table) recall 1.00, precision 0.50 — gold pages [69], retrieved pages [1, 2, 3, 4, 5, 6, 66, 67, 68, 69, 70, 71, 72]
