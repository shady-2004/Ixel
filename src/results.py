
import os
import csv
import json
from pathlib import Path
from typing import List

from evaluate import Evaluator, EvalResult


class ResultsReporter:
    """Generates evaluation reports and training curve data."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluation report
    # ------------------------------------------------------------------
    def generate_eval_report(self, db_path: str, log_path: str, model_path: str) -> str:
        """Run evaluation and write a markdown report file."""
        evaluator = Evaluator(db_path, log_path, model_path)
        results = evaluator.evaluate_all()

        report_path = self.results_dir / "evaluation_report.md"

        lines = [
            "# Ixel — Evaluation Report",
            "",
            f"**Database**: `{Path(db_path).name}`",
            f"**Model**: `{Path(model_path).name}`",
            "",
            "## Cost Comparison",
            "",
            "| Strategy | Cost Before | Cost After | Write Overhead | Net Reward |",
            "|----------|------------|------------|----------------|------------|",
        ]

        for r in results:
            lines.append(
                f"| {r.strategy} | {r.total_cost_before:.2f} | {r.total_cost_after:.2f} "
                f"| {r.total_write_overhead:.2f} | {r.net_reward:.2f} |"
            )

        lines.extend([
            "",
            "## Action Distribution",
            "",
            "| Strategy | INDEX | NO_ACTION | DROP |",
            "|----------|-------|-----------|------|",
        ])

        for r in results:
            lines.append(
                f"| {r.strategy} | {r.actions_taken.get('INDEX', 0)} "
                f"| {r.actions_taken.get('NO_ACTION', 0)} "
                f"| {r.actions_taken.get('DROP', 0)} |"
            )

        # Highlight winner
        best = max(results, key=lambda r: r.net_reward)
        lines.extend([
            "",
            f"**Best strategy**: {best.strategy} (net reward: {best.net_reward:.2f})",
        ])

        report_text = "\n".join(lines)
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"Report saved to {report_path}")
        Evaluator.print_report(results)

        return str(report_path)

    # ------------------------------------------------------------------
    # Training reward log (for plotting)
    # ------------------------------------------------------------------
    @staticmethod
    def save_training_log(rewards: List[float], log_path: str):
        """Save per-step rewards to CSV for later plotting."""
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "rolling_avg_50"])
            rolling = []
            for i, r in enumerate(rewards):
                rolling.append(r)
                if len(rolling) > 50:
                    rolling.pop(0)
                avg = sum(rolling) / len(rolling)
                writer.writerow([i + 1, f"{r:.4f}", f"{avg:.4f}"])
        print(f"Training log saved to {log_path}")

    def generate_training_curve_data(self, rewards: List[float]) -> str:
        """Save training rewards and return the file path."""
        csv_path = str(self.results_dir / "training_curve.csv")
        self.save_training_log(rewards, csv_path)
        return csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logs_dir = data_dir / "logs"

    reporter = ResultsReporter(str(data_dir))

    db_path = str(data_dir / "training_db_1.db")
    log_path = str(logs_dir / "training_db_1.jsonl")
    model_path = str(data_dir / "model.pth")

    if os.path.exists(model_path):
        reporter.generate_eval_report(db_path, log_path, model_path)
    else:
        print("No trained model found. Run train.py first.")
