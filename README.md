# Ixel — Schema-Agnostic RL Index Tuner for SQLite

Ixel is a reinforcement learning agent that learns which columns in a SQLite
database should be indexed, based on how the database is actually queried —
and it works on databases it has never seen before, regardless of schema
shape, column count, or size.

## Benchmark: generalization to an unseen database

The agent was trained exclusively on a `train` database (62,500 rows, 50,000
logged queries) and evaluated, without any further learning, on a completely
independent `test` database with a different schema.

| Strategy | Cost Before | Cost After | Write Overhead | Net Reward |
|----------|------------|------------|----------------|------------|
| No Indexes | 2,637,296.95 | 2,670,259.00 | 0.00 | -32,962.05 |
| Index Everything | 2,703,221.05 | 1,642,325.85 | 389,435.00 | +1,021,951.70 |
| Human Heuristic | 2,769,145.15 | 2,540,705.55 | 167,105.00 | +211,729.10 |
| **RL Policy (Ixel)** | **2,736,183.10** | **1,475,215.85** | **222,325.00** | **+1,238,734.75** |

Ixel selected exactly 6 indexes — fewer than "Index Everything" — and beat
every baseline on net reward, because it accounts for write overhead rather
than just minimizing read cost.

Run it yourself (this will train a new model from scratch):
```bash
pip install -e .
python scripts/run_universal_generalization.py
```

## Production Deployment (Pipeline & Monitor)

While the benchmark proves Ixel's Zero-Shot capabilities, the architecture includes fully-functional modules for live production deployment:

### 1. The Execution Pipeline
You can run the pretrained model directly against your own live database using the provided `run_demo.py` script:
```bash
python scripts/run_demo.py --db-path data/mydb.db --log-path data/logs/mydb.jsonl --model-path models/universal_zero_shot.pth
```
The **Pipeline** takes full control:
- It extracts feature vectors from your live query logs.
- It asks the trained policy whether to `INDEX` or `DROP` a column.
- It executes `CREATE INDEX` on the live database.
- It safely records the original schema in a rollback log when running `DROP INDEX`.

### 3. Custom Training (Optional)
Because Ixel is a **Zero-Shot** model, you do not *need* to train it on your own database. The pre-trained `models/universal_zero_shot.pth` model will instantly generalize to your schema. However, if you want to experiment with the RL environment yourself, you can generate a synthetic workload and train a new model from scratch:
```bash
python scripts/synthetic_workload.py --data-dir ./data
python scripts/train.py --data-dir ./data
```

### 4. The Shift Monitor
The `Monitor` is designed to run as a lightweight background cron job. Instead of relying purely on performance degradation, the monitor directly compares the statistical distribution of column queries (e.g., WHERE/JOIN frequencies). If the workload shifts dramatically (e.g., users stop querying `region` and start querying `signup_date`), the Monitor detects it *before* it causes massive slowdowns and can automatically trigger the Pipeline to re-tune the indexes.

## The problem

Choosing which columns to index is normally either manual DBA judgment or a
fixed rule-of-thumb (index every foreign key, index anything in a WHERE
clause). Neither approach adapts automatically as query patterns change, and
neither generalizes: a rule tuned for one schema doesn't transfer to another
without re-tuning.

Ixel treats this as a learning problem: given a column's structural
properties (type, cardinality, nulls) and how it's actually used in the
query log (WHERE/JOIN/ORDER BY/GROUP BY frequency), predict whether indexing,
dropping, or leaving it alone will improve real query cost — and validate
that prediction before ever touching a live database.

## How it generalizes across arbitrary schemas

The core design choice: the model never looks at an entire database at
once. It looks at **one column at a time**, always represented as the same
fixed-length feature vector (16 numbers), regardless of how many columns or
tables the database actually has.

```
Database (any schema) → one row per column → same shared network,
                                                called once per column
                                              → per-column decision
```

Because the network's input is always the same shape, the exact same
trained weights apply unchanged whether the database has 5 columns or 500 —
there's no dependence on schema size or column names. This is the same
principle behind Deep Sets and per-candidate scoring in recommender systems:
apply one small shared model independently to each item in a variable-size
collection, rather than trying to encode the whole collection into one
fixed-size input.

## The learning algorithm: a contextual bandit, not a full DQN

Each column's indexing decision is evaluated independently against the
current workload snapshot — deciding to index column A doesn't change
whether column B should be indexed, and there's no multi-step sequence of
decisions whose value depends on future outcomes (unlike, say, a game where
one move sets up a later one). Because of this, the model is trained as a
**contextual bandit**: a small neural network predicts the expected reward
of each action (`INDEX`, `NO_ACTION`, `DROP`) for a given column, trained
with plain regression (MSE loss, Adam optimizer) against the reward actually
observed after the action is tried — no discount factor, no bootstrapped
future-state value, no target network. This is a deliberate simplification
of full Q-learning/DQN, chosen because the extra machinery DQN provides
(reasoning about delayed, sequential consequences) doesn't apply at this
scope. See [DESIGN.md](DESIGN.md) for the full reasoning, including where that would
change (e.g. composite-index interactions, multi-cycle tuning budgets).

## How a decision gets validated before being trusted

Nothing the policy predicts is applied blindly:

1. The policy proposes an action for a column, based on its current belief
2. The action is applied to a **scratch copy** of the database, never the
   live one
3. `ANALYZE` is run so SQLite's query planner has real cardinality
   statistics — without this, SQLite will use any available index whether
   or not it's actually beneficial, making it impossible to tell a good
   index from a useless one
4. Cost is measured via `EXPLAIN QUERY PLAN` (a full table `SCAN` is
   penalized heavily, an index `SEARCH` cheaply) — chosen over raw
   wall-clock timing because it's deterministic and unaffected by OS page
   caching, which otherwise makes repeated timing measurements noisy and
   hard to compare fairly
5. The resulting reward (cost improvement, minus a penalty for the extra
   write overhead an index adds) is what the policy is actually trained on
   — only if the reward is positive does a change get applied to the real
   database

## Removing indexes, not just adding them

An index that was useful when added can become dead weight later, if the
query workload shifts. Ixel's action space includes `DROP`, and the
background monitor doesn't only watch for performance degradation — it also
directly compares the *distribution* of what's being queried now against
what was being queried at the last tuning cycle (WHERE/JOIN/ORDER BY/GROUP
BY frequency per column), so a workload shift can be detected even before it
causes a measurable slowdown.

## Known limitations

- **Single-column indexing only** — no composite/multi-column index support
  yet, since the per-column state representation doesn't capture
  cross-column co-occurrence
- **Bandit, not sequential RL** — deliberately doesn't model interactions
  between multiple indexing decisions over time (see [DESIGN.md](DESIGN.md))
- **EXPLAIN QUERY PLAN cost is a proxy**, not real execution time — a
  secondary real-timing check exists but is not the primary reward signal

## License

MIT