# Retriever baseline — golden/tc_u5.jsonl

- run: 20260929-214421 · judge `groq/openai/gpt-oss-120b` · retriever search_document baseline {'k': 3, 'window': 1, 'neighbor_min_score': 0.3}
- index: {'TC-U-5-ONE-SHOT-Notes.pdf': {'chunks': 28, 'fingerprint': 'af70e47cf8fe', 'incomplete': True, 'cache_version': 'v8-code'}}
- search_document match: 4/4

| group | n | recall | sufficient | precision (sent order) | precision (score order) | P@k | 1st useful rank | chunks | words |
|---|---|---|---|---|---|---|---|---|---|
| **in-document** | 4 | 1.00 | 100% | 0.57 | 0.91 | 0.53 | 2.50 | 5.75 | 1125.75 |
| definition | 4 | 1.00 | 100% | 0.57 | 0.91 | 0.53 | 2.50 | 5.75 | 1125.75 |

## Out-of-document questions

In-document top-1 similarity: min 0.69, median 0.73, max 0.80


## Weakest in-document questions

- **tc_u5-004** (definition) recall 1.00, precision 0.33 — gold pages [17], retrieved pages [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
- **tc_u5-002** (definition) recall 1.00, precision 0.33 — gold pages [11], retrieved pages [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16]
- **tc_u5-003** (definition) recall 1.00, precision 0.64 — gold pages [14], retrieved pages [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- **tc_u5-001** (definition) recall 1.00, precision 1.00 — gold pages [5], retrieved pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20]
