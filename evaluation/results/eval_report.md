# Evaluation report - Manufacturing Quality SOP Assistant

Embedding model: `huggingface:BAAI/bge-small-en-v1.5`  |  Generator: `extractive`

| Metric | Value |
|---|---|
| questions | 16 |
| passed | 16 |
| routing_accuracy | 1.0 |
| retrieval_hit_at_3 | 1.0 |
| retrieval_mrr | 1.0 |
| citation_correctness | 1.0 |
| grounded_outcome_rate | 1.0 |
| mean_groundedness | 1.0 |
| avg_processing_ms | 32.994 |

| ID | Category | Type (exp -> got) | Status (exp -> got) | Hit@3 | MRR | Citations | Grounded | ms | Result |
|---|---|---|---|---|---|---|---|---|---|
| Q01 | procedure lookup | procedure -> procedure | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 42.8 | PASS |
| Q02 | acceptable evidence | procedure -> procedure | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 29.0 | PASS |
| Q03 | who to inform | procedure -> procedure | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 31.8 | PASS |
| Q04 | defect synonym - scratches | procedure -> procedure | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 36.7 | PASS |
| Q05 | defect synonym - dimensional mismatch | procedure -> procedure | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 36.3 | PASS |
| Q06 | defect synonym - missing label | procedure -> procedure | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 34.8 | PASS |
| Q07 | defect synonym - failed calibration | procedure -> procedure | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 37.6 | PASS |
| Q08 | procedure comparison | comparison -> comparison | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 49.1 | PASS |
| Q09 | missing tolerance | limit_or_approval -> limit_or_approval | not_verifiable -> not_verifiable | n/a | n/a | PASS | PASS (1.0) | 30.5 | PASS |
| Q10 | contradictory instruction - scratch limit | limit_or_approval -> limit_or_approval | conflict -> conflict | PASS | 1.0 | PASS | PASS (1.0) | 35.2 | PASS |
| Q11 | contradictory instruction - calibration interval | limit_or_approval -> limit_or_approval | conflict -> conflict | PASS | 1.0 | PASS | PASS (1.0) | 28.0 | PASS |
| Q12 | release request (guardrail) | release_action -> release_action | action_refused -> action_refused | PASS | 1.0 | PASS | PASS (1.0) | 25.1 | PASS |
| Q13 | documented limit (verified) | limit_or_approval -> limit_or_approval | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 27.0 | PASS |
| Q14 | approval rule (documented) | limit_or_approval -> limit_or_approval | grounded -> grounded | PASS | 1.0 | PASS | PASS (1.0) | 28.7 | PASS |
| Q15 | missing approval rule | limit_or_approval -> limit_or_approval | not_verifiable -> not_verifiable | n/a | n/a | PASS | PASS (1.0) | 26.8 | PASS |
| Q16 | out of scope | procedure -> procedure | not_verifiable -> not_verifiable | n/a | n/a | PASS | PASS (1.0) | 28.5 | PASS |
