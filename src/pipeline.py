import json
import time
import sqlite3
import contextlib
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from features import FeatureExtractor, ColumnFeatureRow, FEATURE_COUNT
from policy import BanditPolicy
from simulator import Simulator, SimulationResult


@dataclass
class AppliedChange:
    """Record of a single index change applied to the live DB."""
    table: str
    column: str
    action: str         # INDEX | DROP | NO_ACTION
    reward: float
    timestamp: float
    index_name: str     # the actual index name created/dropped
    original_index_sql: Optional[str] = None


class Pipeline:
    """
    Orchestrates: features → policy → simulator → apply to real DB.
    """

    def __init__(self, db_path: str, log_path: str, model_path: str,
                 reward_threshold: float = 0.0):
        self.db_path = db_path
        self.log_path = log_path
        self.model_path = model_path
        self.reward_threshold = reward_threshold

        self.extractor = FeatureExtractor(db_path, log_path)
        self.simulator = Simulator(db_path, log_path)
        self.policy = BanditPolicy(input_dim=FEATURE_COUNT, epsilon=0.0)
        self.policy.load(model_path)

        self.changes: List[AppliedChange] = []

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def run(self) -> List[AppliedChange]:
        """
        Full pipeline pass:
          1. Extract features
          2. For each column, get the policy's action
          3. Simulate each action
          4. Apply only beneficial actions to the live DB
          5. Return the list of changes made
        """
        state = self.extractor.build_state()
        print(f"Pipeline: {len(state)} columns in {Path(self.db_path).name}")

        proposed = []
        for row in state:
            action, q_values = self.policy.select_action(row.vector, explore=False)
            if action == "NO_ACTION":
                continue
            # Simulate to check if this action is actually beneficial
            result = self.simulator.evaluate(row.table, row.column, action)
            proposed.append((row, action, result))

        # Filter to only positive-reward changes
        beneficial = [(row, action, result) for row, action, result in proposed
                      if result.reward > self.reward_threshold]

        print(f"Pipeline: {len(proposed)} actions proposed, {len(beneficial)} beneficial (reward > {self.reward_threshold})")

        # Apply beneficial changes to the real DB
        self.changes = []
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            for row, action, result in beneficial:
                change = self._apply_to_live(conn, row.table, row.column, action, result.reward, result.original_index_sql)
                if change:
                    self.changes.append(change)
            conn.commit()

        self._save_changelog()
        return self.changes

    # ------------------------------------------------------------------
    # Apply to live DB
    # ------------------------------------------------------------------
    def _apply_to_live(self, conn: sqlite3.Connection, table: str, column: str,
                       action: str, reward: float, original_index_sql: Optional[str] = None) -> Optional[AppliedChange]:
        """Apply a single INDEX or DROP to the live database."""
        index_name = f"idx_{table}_{column}_auto"

        try:
            if action == "INDEX":
                conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}"("{column}")')
                print(f"  + Created index {index_name}")
            elif action == "DROP":
                # Find the actual index name covering this column
                actual_name = self._find_index_name(conn, table, column)
                if actual_name:
                    conn.execute(f'DROP INDEX IF EXISTS "{actual_name}"')
                    index_name = actual_name
                    print(f"  - Dropped index {index_name}")
                else:
                    return None
            else:
                return None
        except sqlite3.Error as e:
            print(f"  ! Failed to apply {action} on {table}.{column}: {e}")
            return None

        return AppliedChange(
            table=table,
            column=column,
            action=action,
            reward=reward,
            timestamp=time.time(),
            index_name=index_name,
            original_index_sql=original_index_sql,
        )

    def _find_index_name(self, conn: sqlite3.Connection, table: str, column: str) -> Optional[str]:
        """Find the index name for a given column, if any."""
        indexes = conn.execute(f'PRAGMA index_list("{table}")').fetchall()
        for idx in indexes:
            idx_name = idx["name"]
            cols = conn.execute(f'PRAGMA index_info("{idx_name}")').fetchall()
            for c in cols:
                if c["name"] == column:
                    return idx_name
        return None

    # ------------------------------------------------------------------
    # Changelog for rollback
    # ------------------------------------------------------------------
    def _save_changelog(self):
        """Persist applied changes to a JSON file for auditability and rollback."""
        changelog_path = Path(self.db_path).parent / "changelog.json"

        existing = []
        if changelog_path.exists():
            try:
                with open(changelog_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        for c in self.changes:
            existing.append({
                "table": c.table,
                "column": c.column,
                "action": c.action,
                "reward": c.reward,
                "timestamp": c.timestamp,
                "index_name": c.index_name,
                "original_index_sql": c.original_index_sql,
                "db_path": self.db_path,
            })

        with open(changelog_path, "w") as f:
            json.dump(existing, f, indent=2)

        if self.changes:
            print(f"Pipeline: {len(self.changes)} changes logged to {changelog_path}")

    def rollback(self, last_n: int = 1):
        """
        Undo the most recent changes by reversing the actions.
        """
        changelog_path = Path(self.db_path).parent / "changelog.json"
        existing = []
        if changelog_path.exists():
            try:
                with open(changelog_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
                
        to_rollback = existing if last_n is None else existing[-last_n:]

        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            for change in reversed(to_rollback):
                try:
                    if change["action"] == "INDEX":
                        conn.execute(f'DROP INDEX IF EXISTS "{change["index_name"]}"')
                        print(f"  Rollback: dropped {change['index_name']}")
                    elif change["action"] == "DROP":
                        original_sql = change.get("original_index_sql")
                        if original_sql:
                            conn.execute(original_sql)
                            print(f"  Rollback: re-created {change['index_name']} via original SQL")
                        else:
                            conn.execute(
                                f'CREATE INDEX IF NOT EXISTS "{change["index_name"]}" '
                                f'ON "{change["table"]}"("{change["column"]}")'
                            )
                            print(f"  Rollback: re-created {change['index_name']} (fallback)")
                except sqlite3.Error as e:
                    print(f"  Rollback failed for {change['index_name']}: {e}")
            conn.commit()

        if to_rollback:
            remaining = existing[:-len(to_rollback)] if len(to_rollback) < len(existing) else []
            with open(changelog_path, "w") as f:
                json.dump(remaining, f, indent=2)

    @staticmethod
    def print_changes(changes: List[AppliedChange]):
        if not changes:
            print("Pipeline: no changes applied.")
            return
        print(f"\n{'Action':<10} {'Table':<20} {'Column':<20} {'Index Name':<30} {'Reward':>10}")
        print("-" * 95)
        for c in changes:
            print(f"{c.action:<10} {c.table:<20} {c.column:<20} {c.index_name:<30} {c.reward:>10.2f}")


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logs_dir = data_dir / "logs"

    db_path = str(data_dir / "training_db_1.db")
    log_path = str(logs_dir / "training_db_1.jsonl")
    model_path = str(data_dir / "model.pth")

    if not os.path.exists(model_path):
        print("No trained model found. Run train.py first.")
    else:
        pipe = Pipeline(db_path, log_path, model_path)
        changes = pipe.run()
        Pipeline.print_changes(changes)
