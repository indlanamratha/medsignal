# MedSignal AI Evaluation

## Retrieval (32 label questions, top 5)

| mode                               |   hit_rate_at_5 |   mrr_at_5 |
|:-----------------------------------|----------------:|-----------:|
| bm25                               |           0.844 |      0.776 |
| dense                              |           0.844 |      0.797 |
| hybrid                             |           0.906 |      0.802 |
| hybrid_rerank                      |           0.906 |      0.906 |
| bm25 + expansion + dedupe          |           0.875 |      0.859 |
| dense + expansion + dedupe         |           0.906 |      0.839 |
| hybrid + expansion + dedupe        |           1     |      0.865 |
| hybrid_rerank + expansion + dedupe |           1     |      0.944 |

## Grounded answering (hybrid + rerank)

|                                           | value      |
|:------------------------------------------|:-----------|
| Label questions answered                  | 100% of 32 |
| Fact recall (answered questions)          | 1.00       |
| Faithfulness (LLM judge)                  | 1.00       |
| Unanswerable questions correctly declined | 100% of 10 |

## Agent data questions

Correct: 8 of 8 (100%)
