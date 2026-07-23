import sqlite3
import shutil
import json
import time
import statistics
import os
import tempfile
from typing import List, Optional, Tuple
from dataclasses import dataclass

from query_parser import QueryParser


SCAN_PENALTY = 100.0
SEARCH_PENALTY = 1.0


@dataclass
class SimulationResult:
    action: str
    table: str
    column: str
    cost_before: float
    cost_after: float
    write_overhead: float
    reward: float
    original_index_sql: Optional[str] = None


class Simulator:
    _global_log_cache = {}
    _global_relevant_cache = {}
    _global_write_cache = {}

    def __init__(self, db_path: str, log_path: str, overhead_weight: float = 0.1):
        self.db_path = db_path
        self.log_path = log_path
        self.overhead_weight = overhead_weight
        self.query_parser = QueryParser()

    # ------------------------------------------------------------------
    # Query selection
    # ------------------------------------------------------------------
    def _read_log(self) -> List[dict]:
        if self.log_path in Simulator._global_log_cache:
            return Simulator._global_log_cache[self.log_path]
        
        entries = []
        relevant_map = {}
        write_map = {}
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("error"):
                        continue
                    
                    sql = entry["sql"]
                    entry["usages"] = self.query_parser.parse(sql)
                    sql_upper = sql.strip().upper()
                    entry["sql_upper"] = sql_upper
                    
                    entries.append(entry)

                    # Pre-index by (table, column)
                    for u in entry["usages"]:
                        tbl = u.table.lower() if u.table else "unknown"
                        col = u.column.lower() if u.column else "unknown"
                        key = (tbl, col)
                        if key not in relevant_map:
                            relevant_map[key] = []
                        relevant_map[key].append(entry)

                    # Pre-index write queries by table
                    if sql_upper.startswith(("INSERT", "UPDATE", "DELETE")):
                        tokens = sql_upper.split()
                        table_name = None
                        if tokens[0] == "INSERT" and len(tokens) >= 3:
                            table_name = tokens[2]
                        elif tokens[0] == "UPDATE" and len(tokens) >= 2:
                            table_name = tokens[1]
                        elif tokens[0] == "DELETE" and len(tokens) >= 3:
                            table_name = tokens[2]
                        if table_name:
                            tbl_clean = table_name.strip('"\'()').upper()
                            if tbl_clean not in write_map:
                                write_map[tbl_clean] = []
                            write_map[tbl_clean].append(entry)
        except FileNotFoundError:
            pass
            
        Simulator._global_log_cache[self.log_path] = entries
        Simulator._global_relevant_cache[self.log_path] = relevant_map
        Simulator._global_write_cache[self.log_path] = write_map
        return entries

    def _get_relevant_queries(self, table: str, column: str) -> List[dict]:
        """Queries where (table, column) appears in WHERE/JOIN/ORDER_BY/GROUP_BY."""
        self._read_log()
        relevant_map = Simulator._global_relevant_cache.get(self.log_path, {})
        return relevant_map.get((table.lower(), column.lower()), [])

    def _get_write_queries(self, table: str) -> List[dict]:
        """Logged INSERT/UPDATE/DELETE statements affecting `table`."""
        self._read_log()
        write_map = Simulator._global_write_cache.get(self.log_path, {})
        return write_map.get(table.upper(), [])

    # ------------------------------------------------------------------
    # In-memory scratch copy management
    # ------------------------------------------------------------------
    def _make_scratch_conn(self) -> sqlite3.Connection:
        src = sqlite3.connect(self.db_path)
        mem_conn = sqlite3.connect(":memory:")
        src.backup(mem_conn)
        src.close()
        mem_conn.row_factory = sqlite3.Row
        return mem_conn

    # ------------------------------------------------------------------
    # Applying the action for real
    # ------------------------------------------------------------------
    def _find_index_name(self, conn: sqlite3.Connection, table: str, column: str) -> Optional[str]:
        """Locate the name of an existing index covering this column, if any."""
        index_rows = conn.execute(f'PRAGMA index_list("{table}")').fetchall()
        for idx in index_rows:
            index_name = idx["name"] if isinstance(idx, sqlite3.Row) else idx[1]
            col_rows = conn.execute(f'PRAGMA index_info("{index_name}")').fetchall()
            for c in col_rows:
                col_name = c["name"] if isinstance(c, sqlite3.Row) else c[2]
                if col_name == column:
                    return index_name
        return None

    def _apply_action(self, conn: sqlite3.Connection, table: str, column: str, action: str) -> Optional[str]:
        if action == "INDEX":
            index_name = f"idx_{table}_{column}_auto"
            conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}"("{column}")')
            return None
        elif action == "DROP":
            existing = self._find_index_name(conn, table, column)
            if existing:
                sql_row = conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name = ?", (existing,)).fetchone()
                original_sql = sql_row["sql"] if sql_row else None
                conn.execute(f'DROP INDEX IF EXISTS "{existing}"')
                return original_sql
            return None
        elif action == "NO_ACTION":
            return None
        else:
            raise ValueError(f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # Cost measurement via EXPLAIN QUERY PLAN
    # ------------------------------------------------------------------
    def _estimate_cost(self, conn: sqlite3.Connection, queries: List[dict]) -> float:
        if not queries:
            return 0.0
        
        write_cost = 0.0
        read_cost = 0.0
        read_queries = []
        write_counts = {}

        # 1. Separate reads from writes and group writes by table
        for entry in queries:
            sql_upper = entry.get("sql_upper") or entry["sql"].strip().upper()
            if sql_upper.startswith(("INSERT", "UPDATE", "DELETE")):
                tokens = sql_upper.split()
                table_name = None
                if tokens[0] == "INSERT" and len(tokens) >= 3:
                    table_name = tokens[2]
                elif tokens[0] == "UPDATE" and len(tokens) >= 2:
                    table_name = tokens[1]
                elif tokens[0] == "DELETE" and len(tokens) >= 3:
                    table_name = tokens[2]
                
                if table_name:
                    table_name = table_name.strip('"\'()')
                    write_counts[table_name] = write_counts.get(table_name, 0) + 1
            else:
                read_queries.append(entry)

        # 2. Extremely fast write penalty calculation (1 query per table instead of thousands)
        for table_name, count in write_counts.items():
            try:
                index_count = len(conn.execute(f"PRAGMA index_list('{table_name}')").fetchall())
                write_cost += (index_count * 5.0) * count  # WRITE_PENALTY
            except sqlite3.Error:
                pass

        # 3. Fast random sampling for read queries (Max 100 to prevent freezing)
        import random
        sampled_reads = read_queries
        if len(read_queries) > 100:
            sampled_reads = random.sample(read_queries, 100)
            
        for entry in sampled_reads:
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
                read_cost += SCAN_PENALTY
            elif "SEARCH" in plan_text:
                read_cost += SEARCH_PENALTY
        
        # Scale the read cost back up to match the full log magnitude
        if len(sampled_reads) > 0:
            read_cost = (read_cost / len(sampled_reads)) * len(read_queries)
            
        return read_cost + write_cost

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _measure_real_latency(self, conn: sqlite3.Connection, queries: List[dict], repeats: int = 3) -> float:
        total = 0.0
        for entry in queries:
            sql = entry["sql"]
            params = entry.get("params") or []
            params = [p for p in params if isinstance(p, (int, float, str, bytes))]
            times = []
            for _ in range(repeats):
                try:
                    start = time.perf_counter()
                    conn.execute(sql, params).fetchall()
                    times.append(time.perf_counter() - start)
                except sqlite3.Error:
                    break
            if times:
                total += statistics.median(times)
        return total

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def evaluate(self, table: str, column: str, action: str) -> SimulationResult:
        relevant_queries = self._get_relevant_queries(table, column)
        write_queries = self._get_write_queries(table)

        conn = self._make_scratch_conn()

        try:
            conn.execute("ANALYZE")
            cost_before = self._estimate_cost(conn, relevant_queries)
            write_cost_before = self._estimate_cost(conn, write_queries)

            original_sql = self._apply_action(conn, table, column, action)
            conn.commit()
            conn.execute("ANALYZE")  

            cost_after = self._estimate_cost(conn, relevant_queries)
            write_cost_after = self._estimate_cost(conn, write_queries)

            write_overhead = max(0.0, write_cost_after - write_cost_before)
            reward = (cost_before - cost_after) - (self.overhead_weight * write_overhead)

        finally:
            conn.close()

        return SimulationResult(
            action=action, table=table, column=column,
            cost_before=cost_before, cost_after=cost_after,
            write_overhead=write_overhead, reward=reward,
            original_index_sql=original_sql
        )

    def evaluate_batch(self, actions: List[Tuple[str, str, str]]) -> SimulationResult:
        """Evaluates a batch of (table, column, action) tuples, deduplicating queries."""
        # For a full batch evaluation of a strategy, we must measure against the entire workload
        # to ensure fair cost comparison across different strategies.
        relevant_queries = self._read_log()
        write_queries = [q for q in relevant_queries if q["sql"].strip().upper().startswith(("INSERT", "UPDATE", "DELETE"))]

        conn = self._make_scratch_conn()

        try:
            conn.execute("ANALYZE")
            cost_before = self._estimate_cost(conn, relevant_queries)
            write_cost_before = self._estimate_cost(conn, write_queries)

            for table, column, action in actions:
                self._apply_action(conn, table, column, action)
            conn.commit()
            conn.execute("ANALYZE")

            cost_after = self._estimate_cost(conn, relevant_queries)
            write_cost_after = self._estimate_cost(conn, write_queries)

            write_overhead = max(0.0, write_cost_after - write_cost_before)
            reward = (cost_before - cost_after) - (self.overhead_weight * write_overhead)
        finally:
            conn.close()

        return SimulationResult(
            action="BATCH", table="BATCH", column="BATCH",
            cost_before=cost_before, cost_after=cost_after,
            write_overhead=write_overhead, reward=reward,
            original_index_sql=None
        )


if __name__ == "__main__":
    import sqlite3 as sq
    from logger import LoggedConnection

    db_path = os.path.join(tempfile.gettempdir(), "simulator_demo.db")
    log_path = os.path.join(tempfile.gettempdir(), "simulator_demo.jsonl")
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(log_path):
        os.remove(log_path)

    setup = sq.connect(db_path)
    setup.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    setup.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, status TEXT)")
    setup.executemany("INSERT INTO users (name) VALUES (?)", [(f"user{i}",) for i in range(500)])
    
    setup.executemany(
        "INSERT INTO orders (user_id, status) VALUES (?, ?)",
        [(i % 500, "shipped" if i % 2 == 0 else "pending") for i in range(3000)]
    )
    setup.commit()
    setup.close()

    conn = LoggedConnection(db_path, log_path=log_path)
    for i in range(50):
        conn.execute("SELECT * FROM orders WHERE user_id = ?", (i % 500,))
    for i in range(50):
        conn.execute("SELECT * FROM orders WHERE status = ?", ("shipped",))
    conn.close()

    sim = Simulator(db_path, log_path)

    result_good = sim.evaluate("orders", "user_id", "INDEX")
    print("Indexing HIGH-cardinality column (user_id):")
    print(f"  cost_before={result_good.cost_before}, cost_after={result_good.cost_after}, "
          f"write_overhead={result_good.write_overhead}, reward={result_good.reward}")

    result_bad = sim.evaluate("orders", "status", "INDEX")
    print("\nIndexing LOW-cardinality column (status):")
    print(f"  cost_before={result_bad.cost_before}, cost_after={result_bad.cost_after}, "
          f"write_overhead={result_bad.write_overhead}, reward={result_bad.reward}")

    print(f"\nreward(user_id) > reward(status)? {result_good.reward > result_bad.reward}")
