# Retriever baseline — golden/tc_u5.jsonl

- run: 20261002-070849 · judge `groq/openai/gpt-oss-120b` · retriever search_document baseline {'k': 3, 'window': 1, 'neighbor_min_score': 0.3}
- index: {'TC-U-5-ONE-SHOT-Notes.pdf': {'chunks': 28, 'fingerprint': 'af70e47cf8fe', 'incomplete': True, 'cache_version': 'v8-code'}}
- search_document match: 50/50

| group | n | recall | sufficient | precision (sent order) | precision (score order) | P@k | 1st useful rank | chunks | words |
|---|---|---|---|---|---|---|---|---|---|
| **in-document** | 42 | 0.99 | 95% | 0.96 | 0.93 | 0.75 | 1.05 | 4.12 | 796.79 |
| comparison | 5 | 1.00 | 100% | 0.95 | 0.97 | 0.90 | 1.00 | 2.60 | 504.60 |
| definition | 10 | 1.00 | 100% | 0.98 | 0.91 | 0.66 | 1.00 | 4.80 | 924.10 |
| explanation | 8 | 1.00 | 100% | 0.95 | 0.86 | 0.71 | 1.00 | 4.12 | 804.75 |
| figure | 4 | 1.00 | 100% | 1.00 | 1.00 | 0.92 | 1.00 | 2.50 | 479.50 |
| list | 7 | 0.99 | 86% | 0.96 | 0.91 | 0.61 | 1.00 | 5.00 | 967.71 |
| multi-hop | 8 | 0.97 | 88% | 0.91 | 0.97 | 0.86 | 1.25 | 4.25 | 821.38 |

## Out-of-document questions

In-document top-1 similarity: min 0.37, median 0.63, max 0.80

- tc_u5-043: top-1 similarity 0.48, useful chunks 0/6
- tc_u5-044: top-1 similarity 0.26, useful chunks 0/0
- tc_u5-045: top-1 similarity 0.46, useful chunks 0/0
- tc_u5-046: top-1 similarity 0.21, useful chunks 0/0
- tc_u5-047: top-1 similarity 0.44, useful chunks 0/1
- tc_u5-048: top-1 similarity 0.33, useful chunks 0/0
- tc_u5-049: top-1 similarity 0.15, useful chunks 0/0
- tc_u5-050: top-1 similarity 0.11, useful chunks 0/0

## Weakest in-document questions

- **tc_u5-030** (multi-hop) recall 0.73, precision 0.37 — gold pages [31, 32, 33, 35], retrieved pages [1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32, 33, 34, 35, 36]
  - missing: Achievers perform work in a highly creative way.
  - missing: Achievers perform work in an original way.
  - missing: Achievers perform work in an innovative way.
- **tc_u5-014** (list) recall 0.93, precision 1.00 — gold pages [29], retrieved pages [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
  - missing: Perfectionists do not easily show vulnerability.
- **tc_u5-016** (list) recall 1.00, precision 0.70 — gold pages [41], retrieved pages [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56]
- **tc_u5-041** (comparison) recall 1.00, precision 0.75 — gold pages [30, 31], retrieved pages [27, 28, 29, 30, 31, 32, 33, 53, 54, 55, 56]
- **tc_u5-006** (definition) recall 1.00, precision 0.81 — gold pages [28], retrieved pages [25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58]
