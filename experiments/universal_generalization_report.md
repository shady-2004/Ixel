# Ixel - Universal Generalization (Zero-Shot) Report

This report proves the ability of the RL Agent to learn universal indexing rules on one database and successfully apply them to a completely **unseen** database with a totally different randomized data distribution and query workload.

**Training Database size**: 62,500 rows | 50,000 queries
**Evaluation Database size**: 62,500 rows | 50,000 queries
**RL Training Steps**: 15,000

## Performance Enhancement (Costs)

| Strategy | Cost Before | Cost After | Write Overhead | Net Reward |
|----------|------------|------------|----------------|------------|
| No Indexes | 2637296.95 | 2670259.00 | 0.00 | -32962.05 |
| Index Everything | 2703221.05 | 1642325.85 | 389435.00 | 1021951.70 |
| Heuristic (FK + WHERE freq) | 2769145.15 | 2540705.55 | 167105.00 | 211729.10 |
| RL Policy | 2736183.10 | 1475215.85 | 222325.00 | 1238734.75 |

## Action Distribution (Model Accuracy / Choices)
Shows how many times the strategy chose to INDEX vs DROP vs NO_ACTION.

| Strategy | INDEX | NO_ACTION | DROP |
|----------|-------|-----------|------|
| No Indexes | 0 | 0 | 12 |
| Index Everything | 12 | 0 | 0 |
| Heuristic (FK + WHERE freq) | 5 | 7 | 0 |
| RL Policy | 6 | 6 | 0 |

## Conclusion
**Best Performing Strategy**: RL Policy
**Peak Net Reward**: 1238734.75

Detailed training trajectory (rewards and penalties over time) has been saved to `experiments/massive_training_curve.csv`.