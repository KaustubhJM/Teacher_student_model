# Retriever baseline — golden/dbms_u5.jsonl

- run: 20260929-023354 · judge `groq/openai/gpt-oss-120b` · retriever search_document baseline {'k': 3, 'window': 1, 'neighbor_min_score': 0.3}
- index: {'DBMS_U_5_ONE_SHOT.pdf': {'chunks': 38, 'fingerprint': 'f2c51b079de7', 'incomplete': True, 'cache_version': 'v8-code'}}
- search_document match: 2/2

| group | n | recall | sufficient | precision (sent order) | precision (score order) | P@k | 1st useful rank | chunks | words |
|---|---|---|---|---|---|---|---|---|---|
| **in-document** | 2 | 1.00 | 100% | 0.85 | 0.96 | 0.86 | 1.50 | 5.50 | 1082.50 |
| definition | 2 | 1.00 | 100% | 0.85 | 0.96 | 0.86 | 1.50 | 5.50 | 1082.50 |

## Out-of-document questions

In-document top-1 similarity: min 0.71, median 0.71, max 0.71


## Weakest in-document questions

- **dbms_u5-002** (definition) recall 1.00, precision 0.71 — gold pages [13], retrieved pages [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
- **dbms_u5-001** (definition) recall 1.00, precision 1.00 — gold pages [4], retrieved pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
