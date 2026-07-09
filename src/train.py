import os
import random
import glob
from pathlib import Path
from collections import deque
import statistics

from features import FeatureExtractor
from policy import BanditPolicy
from simulator import Simulator
from results import ResultsReporter

DATA_DIR = Path(r"d:\ixel\Ixel\data")
LOGS_DIR = DATA_DIR / "logs"

class Trainer:
    def __init__(self, data_dir: str = str(DATA_DIR), logs_dir: str = str(LOGS_DIR)):
        self.data_dir = Path(data_dir)
        self.logs_dir = Path(logs_dir)
        self.dbs = self._get_training_dbs()
        self.db_states = {}
        self.reward_history = []
        
        self.input_dim = 16 
        self.policy = BanditPolicy(input_dim=self.input_dim, lr=1e-3, epsilon=1.0)
        
    def _get_training_dbs(self):
        db_files = glob.glob(str(self.data_dir / "training_db_*.db"))
        dbs = []
        for db_path in db_files:
            db_name = Path(db_path).stem
            log_path = self.logs_dir / f"{db_name}.jsonl"
            if log_path.exists():
                dbs.append((db_path, str(log_path)))
        return dbs

    def prepare(self):
        if not self.dbs:
            print("No training databases found. Run synthetic_workload.py first.")
            return False

        print(f"Found {len(self.dbs)} training databases.")

        # Pre-extract features for all DBs
        print("Pre-extracting features for all DBs...")
        for db_path, log_path in self.dbs:
            extractor = FeatureExtractor(db_path, log_path)
            self.db_states[(db_path, log_path)] = extractor.build_state()
            print(f"  Extracted {len(self.db_states[(db_path, log_path)])} column features for {Path(db_path).name}")
        return True

    def train(self, num_steps: int = 1000) -> list:
        if not self.db_states:
            if not self.prepare():
                return []

        self.reward_history = []

        initial_epsilon = 1.0
        final_epsilon = 0.1
        decay_steps = int(num_steps * 0.7)

        rolling_rewards = deque(maxlen=50)
        
        print("\nStarting training loop...")
        for step in range(1, num_steps + 1):
            if step <= decay_steps:
                self.policy.epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (step / decay_steps)
            else:
                self.policy.epsilon = final_epsilon

            db_path, log_path = random.choice(self.dbs)
            state_rows = self.db_states[(db_path, log_path)]
            
            if not state_rows:
                continue
                
            chosen_row = random.choice(state_rows)
            
            action, q_values = self.policy.select_action(chosen_row.vector, explore=True)
            
            sim = Simulator(db_path, log_path)
            sim_result = sim.evaluate(chosen_row.table, chosen_row.column, action)
            
            loss = self.policy.update(chosen_row.vector, action, sim_result.reward)
            
            rolling_rewards.append(sim_result.reward)
            self.reward_history.append(sim_result.reward)
            
            if step % 50 == 0:
                avg_reward = statistics.mean(rolling_rewards) if rolling_rewards else 0.0
                print(f"Step {step:4d}/{num_steps} | Epsilon: {self.policy.epsilon:.2f} | Loss: {loss:.4f} | Avg Reward (last 50): {avg_reward:.4f}")

        model_path = self.data_dir / "model.pth"
        self.policy.save(str(model_path))
        print(f"\nTraining complete. Policy saved to {model_path}")

        # Save training curve data
        reporter = ResultsReporter(str(self.data_dir))
        reporter.generate_training_curve_data(self.reward_history)

        return self.reward_history

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(num_steps=1000)
