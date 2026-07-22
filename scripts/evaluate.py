import os
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from features import FeatureExtractor, ColumnFeatureRow, FEATURE_COUNT
from policy import BanditPolicy, ACTIONS
from simulator import Simulator, SimulationResult


@dataclass
class EvalResult:
    strategy: str
    total_cost_before: float
    total_cost_after: float
    total_write_overhead: float
    net_reward: float
    actions_taken: Dict[str, int]   # {"INDEX": n, "DROP": n, "NO_ACTION": n}


class Evaluator:
    """
    Runs a trained policy and multiple baselines on a held-out database,
    producing a comparison table of cost reduction.
    """

    def __init__(self, db_path: str, log_path: str, model_path: str):
        self.db_path = db_path
        self.log_path = log_path
        self.model_path = model_path

        self.extractor = FeatureExtractor(db_path, log_path)
        self.simulator = Simulator(db_path, log_path)
        self.state: List[ColumnFeatureRow] = []

    def prepare(self):
        """Extract features for every column in the held-out DB."""
        self.state = self.extractor.build_state()
        print(f"Evaluator: extracted {len(self.state)} columns from {Path(self.db_path).name}")

    # ------------------------------------------------------------------
    # Strategy runners
    # ------------------------------------------------------------------
    def _run_strategy(self, strategy_name: str, action_fn) -> EvalResult:
        """Generic runner: calls action_fn(row) for each column, simulates batch, totals."""
        actions = []
        action_counts = {a: 0 for a in ACTIONS}

        for row in self.state:
            action = action_fn(row)
            actions.append((row.table, row.column, action))
            action_counts[action] += 1

        result = self.simulator.evaluate_batch(actions)

        return EvalResult(
            strategy=strategy_name,
            total_cost_before=result.cost_before,
            total_cost_after=result.cost_after,
            total_write_overhead=result.write_overhead,
            net_reward=result.reward,
            actions_taken=action_counts,
        )

    def run_policy(self) -> EvalResult:
        """Run the trained RL policy with explore=False."""
        policy = BanditPolicy(input_dim=FEATURE_COUNT, epsilon=0.0)
        policy.load(self.model_path)

        def pick(row: ColumnFeatureRow) -> str:
            action, _ = policy.select_action(row.vector, explore=False)
            return action

        return self._run_strategy("RL Policy", pick)

    def run_no_index(self) -> EvalResult:
        """Baseline: DROP every index (no indexes at all)."""
        return self._run_strategy("No Indexes", lambda row: "DROP")

    def run_index_everything(self) -> EvalResult:
        """Baseline: INDEX every column."""
        return self._run_strategy("Index Everything", lambda row: "INDEX")

    def run_heuristic(self) -> EvalResult:
        """
        Simple heuristic baseline:
          - Index if column is FK  (vector[5] == 1.0)
          - Index if WHERE frequency > 0.05  (vector[10] > 0.05)
          - Otherwise NO_ACTION
        """
        def pick(row: ColumnFeatureRow) -> str:
            is_fk = row.vector[5]
            where_freq = row.vector[10]
            if is_fk == 1.0 or where_freq > 0.05:
                return "INDEX"
            return "NO_ACTION"

        return self._run_strategy("Heuristic (FK + WHERE freq)", pick)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def evaluate_all(self) -> List[EvalResult]:
        """Run all strategies and return the comparison list."""
        if not self.state:
            self.prepare()

        results = [
            self.run_no_index(),
            self.run_index_everything(),
            self.run_heuristic(),
            self.run_policy(),
        ]
        return results

    @staticmethod
    def print_report(results: List[EvalResult]):
        """Pretty-print a comparison table to stdout."""
        print("\n" + "=" * 90)
        print(f"{'Strategy':<30} {'Cost Before':>12} {'Cost After':>12} {'Write OH':>10} {'Net Reward':>12}")
        print("-" * 90)
        for r in results:
            print(f"{r.strategy:<30} {r.total_cost_before:>12.2f} {r.total_cost_after:>12.2f} "
                  f"{r.total_write_overhead:>10.2f} {r.net_reward:>12.2f}")
        print("-" * 90)

        # Action distribution
        print(f"\n{'Strategy':<30} {'INDEX':>8} {'NO_ACTION':>10} {'DROP':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r.strategy:<30} {r.actions_taken.get('INDEX', 0):>8} "
                  f"{r.actions_taken.get('NO_ACTION', 0):>10} {r.actions_taken.get('DROP', 0):>8}")
        print("=" * 90)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logs_dir = data_dir / "logs"

    db_path = str(data_dir / "training_db_1.db")
    log_path = str(logs_dir / "training_db_1.jsonl")
    model_path = str(data_dir / "model.pth")

    if not os.path.exists(model_path):
        print("No trained model found. Run train.py first.")
    else:
        evaluator = Evaluator(db_path, log_path, model_path)
        results = evaluator.evaluate_all()
        Evaluator.print_report(results)
