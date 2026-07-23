# Ixel Design Document

## Core Concepts
Ixel applies a Contextual Bandit approach to the problem of physical database design (specifically, secondary index selection) in SQLite.

Instead of treating the entire database as a single state space (like DQN), Ixel evaluates actions on a *per-column* basis. The state vector characterizes the column (cardinality, null ratio, frequency in WHERE/JOIN/ORDER BY clauses).

## The Reward Function (Read vs. Write Trade-off)
The core mathematical challenge of Ixel is balancing the speed of `SELECT` queries against the slowdown of `INSERT/UPDATE/DELETE` queries. When the `Simulator` tests an action, the Net Reward is calculated as:

$$ Net Reward = (\Delta Read Cost) - (Write Penalty) $$

- **Read Cost Reduction**: Measured via `EXPLAIN QUERY PLAN` on the read-heavy queries in the workload.
- **Write Penalty**: Calculated by counting the exact number of rows modified by the write queries in the workload, multiplied by a strict penalty factor (because every B-Tree write is expensive).

If the model suggests an index that speeds up reads by 10,000 units, but slows down writes by 15,000 units, it receives a negative reward and learns to `DROP` or ignore that index in the future.

## Modules

### 1. Feature Extraction (`features.py`)
Parses recent query logs to build a normalized feature vector for each column.
Key metrics include:
- `cardinality_ratio`: Distinct values / Total rows
- `where_freq`: Frequency of the column appearing in WHERE clauses.

### 2. Simulator (`simulator.py`)
Uses `EXPLAIN QUERY PLAN` on a temporary scratch copy of the database to measure the expected cost reduction (or penalty) of applying an index.
Fixes included query count normalization and batch deduplication to avoid overestimating savings from overlapping queries.

### 3. Policy (`policy.py`)
A Contextual Bandit implemented as a 3-layer PyTorch neural network.
- **Input**: 16-dimensional column feature vector.
- **Output**: Q-values for `INDEX`, `NO_ACTION`, `DROP`.

### 4. Monitor (`monitor.py`)
A lightweight daemon/cron script that checks:
- Cost degradation relative to a baseline.
- Stale indexes (created by Ixel but rarely used).
- Workload shifts (significant changes in the feature vectors of queried columns).

### 5. Pipeline (`pipeline.py`)
Orchestrates the actual deployment of changes to the live database. Supports robust `rollback()` by storing original `CREATE INDEX` SQL statements for any dropped indexes.

---

## Architectural Decisions & Future Expansions

### Why a Contextual Bandit instead of a full DQN?
In standard Reinforcement Learning, agents navigate a Markov Decision Process (MDP) where action $A_t$ changes the state to $S_{t+1}$, which deeply impacts the value of future actions. We deliberately chose **not** to model Ixel this way for single-column indexing.

Deciding to index `customers.region` does not change the structural state or query frequency of `products.category`. Each column's reward can be evaluated mostly independently. Because there are no "delayed, multi-step sequential consequences" (like moving a chess piece to set up a future trap), the heavy machinery of Q-Learning (discount factors, Bellman equations, Target Networks) is unnecessary overhead. A Contextual Bandit is mathematically perfect for independent, stateless decisions.

### Training the Bandit (Epsilon-Greedy Exploration)
The model is trained using an **Epsilon-Greedy** exploration strategy over thousands of steps:
1. **Explore**: With probability $\epsilon$ (starting at 1.0), the agent takes a completely random action (`INDEX`, `DROP`, `NO_ACTION`) to discover how the database reacts.
2. **Exploit**: With probability $1 - \epsilon$, the agent trusts its neural network and takes the action with the highest predicted Net Reward.
3. **Update**: After taking the action, it asks the `Simulator` for the actual Net Reward. It calculates the Mean Squared Error (MSE) between its *predicted* reward and the *actual* reward, and updates its weights using the Adam Optimizer.

As training progresses (e.g., from step 1 to 15,000), $\epsilon$ decays linearly from 1.0 down to 0.1, slowly shifting the model from pure random experimentation to mathematically optimal exploitation.

### When would we need a full MDP / Sequential RL?
If we expand Ixel in the future, we will need to upgrade to a full Sequential RL model (like PPO or DQN) for two specific scenarios:

1. **Composite / Multi-Column Indexes**: 
   If the agent is allowed to build an index on `(region, signup_date)`, then the decision to index `region` *does* impact the state of `signup_date`. The agent would need to learn sequential combinations, placing it firmly in MDP territory.
   
2. **Multi-Cycle Tuning Budgets**:
   If the DBA imposes a strict storage budget (e.g., "You may only create 500MB of indexes"), the agent can no longer evaluate columns independently. Indexing a massive column might consume 400MB of the budget, preventing the agent from indexing 5 smaller, highly-queried columns. This turns the problem into a sequential Knapsack Problem, which requires full RL to learn long-term budget management.
