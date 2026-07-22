import json
import time
import sqlite3
import contextlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from features import FeatureExtractor
from query_parser import QueryParser


@dataclass
class DriftReport:
    """Result of a single drift-detection check."""
    timestamp: float
    baseline_cost: float
    current_cost: float
    drift_ratio: float          # current / baseline  (>1.0 means degradation)
    triggered: bool             # True if drift exceeded threshold
    stale_indexes: List[str]    # index names that appear unused
    workload_shifted: bool = False
    shifted_columns: List[str] = field(default_factory=list)
    trigger_reasons: List[str] = field(default_factory=list)


@dataclass
class IndexUsageInfo:
    """Tracks how much a specific index is referenced in the query log."""
    index_name: str
    table: str
    column: str
    query_hits: int             # how many logged queries touch this column in WHERE/JOIN


class Monitor:
    """
    Watches a live database for performance drift and stale indexes.

    Usage:
        mon = Monitor(db_path, log_path, baseline_cost=120.0)
        report = mon.check()
        if report.triggered:
            pipeline.run(db_path, log_path, model_path)
    """

    def __init__(
        self,
        db_path: str,
        log_path: str,
        baseline_cost: Optional[float] = None,
        drift_threshold: float = 1.2,
        stale_hit_rate_threshold: float = 0.01,
        workload_shift_threshold: float = 0.5,
        window_seconds: float = 3600.0,
    ):
        self.db_path = db_path
        self.log_path = log_path
        self.drift_threshold = drift_threshold
        self.stale_hit_rate_threshold = stale_hit_rate_threshold
        self.workload_shift_threshold = workload_shift_threshold
        self.window_seconds = window_seconds
        self.query_parser = QueryParser()

        # If no baseline supplied, compute one from the full log on first use
        self.baseline_cost = baseline_cost
        self.baseline_features: Dict[tuple, List[float]] = {}

    # ------------------------------------------------------------------
    # Cost measurement
    # ------------------------------------------------------------------
    def _compute_current_cost(self) -> float:
        """
        Sum up EXPLAIN QUERY PLAN costs for recent queries and normalize by count.
        """
        recent_queries = self._get_recent_queries()
        if not recent_queries:
            return 0.0

        total_cost = 0.0

        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("ANALYZE")
            for entry in recent_queries:
                sql = entry["sql"]
                try:
                    params = entry.get("params") or []
                    params = [p for p in params if isinstance(p, (int, float, str, bytes))]
                    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
                except sqlite3.Error:
                    try:
                        rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
                    except sqlite3.Error:
                        continue

                plan_text = " ".join(str(row["detail"]) for row in rows).upper()
                if "SCAN" in plan_text:
                    total_cost += 100.0
                elif "SEARCH" in plan_text:
                    total_cost += 1.0

        return total_cost / len(recent_queries)

    def _get_recent_queries(self) -> List[dict]:
        """Read log entries within the time window."""
        entries = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("error"):
                        continue
                    entries.append(entry)
        except FileNotFoundError:
            return []

        if not entries:
            return []

        # Apply time window
        newest_ts = max(e["timestamp"] for e in entries)
        cutoff = newest_ts - self.window_seconds
        return [e for e in entries if e["timestamp"] >= cutoff]

    # ------------------------------------------------------------------
    # Stale index detection
    # ------------------------------------------------------------------
    def _find_stale_indexes(self) -> List[IndexUsageInfo]:
        """
        For every user-created index in the DB, check how many recent queries
        actually reference the indexed column. If the rate is below the
        stale_hit_rate_threshold, flag it as stale.
        """
        recent_queries = self._get_recent_queries()
        stale = []
        if not recent_queries:
            return []
            
        total_queries = max(len(recent_queries), 1)

        # Count column hits across recent queries
        column_hits: Dict[tuple, int] = {}
        for entry in recent_queries:
            sql = entry["sql"]
            usages = self.query_parser.parse(sql)
            for u in usages:
                if u.table:
                    key = (u.table, u.column)
                    column_hits[key] = column_hits.get(key, 0) + 1

        # Walk all indexes in the DB
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()

            for table_row in tables:
                table = table_row["name"]
                indexes = conn.execute(f'PRAGMA index_list("{table}")').fetchall()
                for idx in indexes:
                    index_name = idx["name"]
                    # Skip auto-indexes created by SQLite for PKs
                    if index_name.startswith("sqlite_autoindex"):
                        continue

                    cols = conn.execute(f'PRAGMA index_info("{index_name}")').fetchall()
                    for col in cols:
                        col_name = col["name"]
                        if col_name is None:
                            continue
                        hits = column_hits.get((table, col_name), 0)
                        rate = hits / total_queries
                        info = IndexUsageInfo(
                            index_name=index_name,
                            table=table,
                            column=col_name,
                            query_hits=hits,
                        )
                        if rate < self.stale_hit_rate_threshold:
                            stale.append(info)

        return stale

    def _compute_workload_shift(self) -> Tuple[bool, List[str], Dict[tuple, List[float]]]:
        extractor = FeatureExtractor(self.db_path, self.log_path, window_seconds=self.window_seconds)
        state = extractor.build_state()
        
        current_features = {}
        shifted_columns = []
        
        for row in state:
            key = (row.table, row.column)
            freqs = row.vector[10:14] # where, join, orderby, groupby
            current_features[key] = freqs
            
            if self.baseline_features and key in self.baseline_features:
                base_freqs = self.baseline_features[key]
                distance = sum(abs(a - b) for a, b in zip(freqs, base_freqs))
                if distance > self.workload_shift_threshold:
                    shifted_columns.append(f"{row.table}.{row.column}")
                    
        is_shifted = len(shifted_columns) > 0
        return is_shifted, shifted_columns, current_features

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------
    def check(self) -> DriftReport:
        """
        Run a single drift-detection + stale-index sweep + workload shift check.
        Returns a DriftReport indicating whether the pipeline should re-run.
        """
        current_cost = self._compute_current_cost()
        is_shifted, shifted_columns, current_features = self._compute_workload_shift()

        # Lazy baseline: first check sets it
        if self.baseline_cost is None:
            self.baseline_cost = max(current_cost, 1.0)
            self.baseline_features = current_features

        drift_ratio = current_cost / max(self.baseline_cost, 1.0)
        stale = self._find_stale_indexes()
        stale_names = [s.index_name for s in stale]

        cost_triggered = drift_ratio >= self.drift_threshold
        stale_triggered = len(stale) > 0
        workload_triggered = is_shifted

        reasons = []
        if cost_triggered: reasons.append("COST_DRIFT")
        if stale_triggered: reasons.append("STALE_INDEXES")
        if workload_triggered: reasons.append("WORKLOAD_SHIFT")

        triggered = cost_triggered or stale_triggered or workload_triggered

        report = DriftReport(
            timestamp=time.time(),
            baseline_cost=self.baseline_cost,
            current_cost=current_cost,
            drift_ratio=drift_ratio,
            triggered=triggered,
            stale_indexes=stale_names,
            workload_shifted=workload_triggered,
            shifted_columns=shifted_columns,
            trigger_reasons=reasons,
        )
        return report

    def update_baseline(self, new_baseline: float, new_features: Dict[tuple, List[float]] = None):
        """Call after a successful pipeline re-tune to reset the baseline."""
        self.baseline_cost = new_baseline
        if new_features:
            self.baseline_features = new_features

    @staticmethod
    def print_report(report: DriftReport):
        print(f"\n--- Drift Report [{time.strftime('%H:%M:%S', time.localtime(report.timestamp))}] ---")
        print(f"  Baseline cost : {report.baseline_cost:.2f}")
        print(f"  Current cost  : {report.current_cost:.2f}")
        print(f"  Drift ratio   : {report.drift_ratio:.3f}  (threshold triggers at >= 1.2)")
        print(f"  Stale indexes : {report.stale_indexes if report.stale_indexes else 'None'}")
        print(f"  Workload shift: {'YES' if report.workload_shifted else 'NO'} {report.shifted_columns if report.workload_shifted else ''}")
        print(f"  Triggered     : {'YES — ' + ', '.join(report.trigger_reasons) if report.triggered else 'No'}")
        print("---")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logs_dir = data_dir / "logs"

    db_path = str(data_dir / "training_db_1.db")
    log_path = str(logs_dir / "training_db_1.jsonl")

    mon = Monitor(db_path, log_path, window_seconds=99999)
    report = mon.check()
    Monitor.print_report(report)
