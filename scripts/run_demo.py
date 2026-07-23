import argparse
import sys
from pathlib import Path

# Add src to sys path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Ixel End-to-End Demo")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--log-path", required=True, help="Path to query log (.jsonl)")
    parser.add_argument("--model-path", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--reward-threshold", type=float, default=0.0, help="Minimum reward to apply change")
    args = parser.parse_args()
    
    print(f"Running Ixel pipeline on {args.db_path}...")
    pipeline = Pipeline(args.db_path, args.log_path, args.model_path, reward_threshold=args.reward_threshold)
    pipeline.run()

if __name__ == "__main__":
    main()
