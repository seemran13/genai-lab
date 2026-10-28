# P5 Chunking Strategy Comparison

- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Documents: 5 (pdf, markdown, text, text, markdown)
- Evaluation: 17 patient-scenario queries, hit if a top-k chunk contains the key fact
- Fixed: 90 words, 20 overlap. Header: max 160, min 25 words

| strategy | chunks | avg_words | max_words | hit@1 | hit@3 | hit@5 | mrr@5 | avg_context_words_top3 |
|---|---|---|---|---|---|---|---|---|
| fixed | 18 | 84.2 | 90 | 0.706 | 1.0 | 1.0 | 0.824 | 255.8 |
| header | 25 | 55.6 | 81 | 0.765 | 1.0 | 1.0 | 0.873 | 166.1 |

**Result on this run:** header-based chunking scored higher. With only 17 queries, differences of one or two queries are within noise; build a larger query set before generalising.

## Queries with no hit in top-5
- **fixed**: none
- **header**: none
