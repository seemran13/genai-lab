# P6 Hybrid RAG Benchmark

- Embedder `sentence-transformers/all-MiniLM-L6-v2`, re-ranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, candidates 20, RRF k 60
- 91 clause chunks from 7 synthetic contracts; 36 queries

| mode | recall@1 | recall@3 | recall@5 | mrr@10 | ndcg@5 | mrr_id | mrr_mix | mrr_sem | ms_per_query |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.667 | 0.75 | 0.806 | 0.739 | 0.741 | 1.0 | 1.0 | 0.373 | 0.4 |
| dense | 0.528 | 0.833 | 0.917 | 0.693 | 0.741 | 0.689 | 0.938 | 0.567 | 13.0 |
| hybrid_rrf | 0.694 | 0.861 | 0.889 | 0.789 | 0.803 | 0.962 | 1.0 | 0.526 | 12.1 |
| hybrid_rerank | 0.861 | 1.0 | 1.0 | 0.921 | 0.941 | 1.0 | 1.0 | 0.811 | 330.5 |

Best MRR@10 on this run: **hybrid_rerank**. With 36 queries, gaps of a few points are within noise.

## Context compression (top-3 after re-ranking)
| sub_clauses_kept | avg_context_words | fact_retained | words_saved_pct |
|---|---|---|---|
| all | 170.7 | 36/36 | 0.0 |
| 2 | 150.9 | 33/36 | 11.6 |
| 1 | 98.1 | 33/36 | 42.5 |
