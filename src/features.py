import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

from schema import SchemaReader, ColumnInfo
from query_parser import QueryParser, ColumnUsage


FEATURE_NAMES = [
    "is_integer", "is_text", "is_real", "is_blob",       
    "is_pk", "is_fk", "is_indexed",                       
    "null_ratio", "cardinality_ratio", "row_count_log",   
    "where_freq", "join_freq", "orderby_freq", "groupby_freq",  
    "equality_op_ratio",                                   
    "avg_latency_norm",                                    
]
FEATURE_COUNT = len(FEATURE_NAMES) 


@dataclass
class ColumnFeatureRow:
    table: str
    column: str
    vector: List[float]  

class FeatureExtractor:
    def __init__(self, db_path: str, log_path: str, window_seconds: float = None):

        self.db_path = db_path
        self.log_path = log_path
        self.window_seconds = window_seconds
        self.schema_reader = SchemaReader(db_path)
        self.query_parser = QueryParser()

    def build_state(self) -> List[ColumnFeatureRow]:
        columns = self.schema_reader.get_all_columns()
        usage_stats, latency_stats, total_queries = self._aggregate_log()

        rows = []
        for col in columns:
            key = (col.table_name, col.column_name)
            usage = usage_stats.get(key, self._empty_usage())
            avg_latency = latency_stats.get(key, 0.0)

            vector = self._build_vector(col, usage, avg_latency, total_queries, latency_stats)
            rows.append(ColumnFeatureRow(table=col.table_name, column=col.column_name, vector=vector))

        return rows

    # ---- log aggregation ----

    def _aggregate_log(self) -> Tuple[Dict, Dict, int]:
        """
        Reads the JSONL log and returns:
          usage_stats:   {(table, col): {"WHERE": n, "JOIN": n, "ORDER_BY": n, "GROUP_BY": n, "EQ": n, "RANGE": n}}
          latency_sums:  {(table, col): [sum_latency, count]}  -> converted to avg at the end
          total_queries: int, denominator for frequency ratios
        """
        usage_stats = defaultdict(self._empty_usage)
        latency_accum = defaultdict(lambda: [0.0, 0])
        total_queries = 0

        try:
            with open(self.log_path, "r") as f:
                raw_lines = [line.strip() for line in f if line.strip()]

            entries = [json.loads(line) for line in raw_lines]

            if self.window_seconds is not None and entries:
                newest_ts = max(e["timestamp"] for e in entries)
                cutoff = newest_ts - self.window_seconds
                entries = [e for e in entries if e["timestamp"] >= cutoff]

            for entry in entries:
                if entry.get("error"):
                    continue  

                total_queries += 1
                sql = entry["sql"]
                latency = entry.get("latency_ms", 0.0)

                usages = self.query_parser.parse(sql)
                touched_columns_this_query = set()

                for usage in usages:
                    if usage.table is None:
                        continue  # unresolved table, skip 
                    key = (usage.table, usage.column)
                    stats = usage_stats[key]
                    stats[usage.clause] += 1
                    if usage.operator in ("EQ",):
                        stats["EQ"] += 1
                    elif usage.operator in ("GT", "GTE", "LT", "LTE", "BETWEEN"):
                        stats["RANGE"] += 1
                    touched_columns_this_query.add(key)

                # attribute this query's latency to every column it touched
                for key in touched_columns_this_query:
                    latency_accum[key][0] += latency
                    latency_accum[key][1] += 1

        except FileNotFoundError:
            # no log yet for this DB -- return empty 
            pass

        latency_avgs = {k: (v[0] / v[1] if v[1] > 0 else 0.0) for k, v in latency_accum.items()}
        return usage_stats, latency_avgs, max(total_queries, 1)  # avoid div-by-zero downstream

    @staticmethod
    def _empty_usage() -> Dict[str, int]:
        return {"WHERE": 0, "JOIN": 0, "ORDER_BY": 0, "GROUP_BY": 0, "EQ": 0, "RANGE": 0}

    # ---- vector assembly ----

    def _build_vector(self, col: ColumnInfo, usage: Dict[str, int], avg_latency: float,
                       total_queries: int, latency_stats: Dict) -> List[float]:
        dtype = col.dtype.upper()
        is_integer = 1.0 if "INT" in dtype else 0.0
        is_text = 1.0 if dtype in ("TEXT", "CHAR", "VARCHAR", "CLOB") else 0.0
        is_real = 1.0 if dtype in ("REAL", "FLOAT", "DOUBLE", "NUMERIC") else 0.0
        is_blob = 1.0 if dtype == "BLOB" else 0.0

        where_freq = usage["WHERE"] / total_queries
        join_freq = usage["JOIN"] / total_queries
        orderby_freq = usage["ORDER_BY"] / total_queries
        groupby_freq = usage["GROUP_BY"] / total_queries

        eq_plus_range = usage["EQ"] + usage["RANGE"]
        equality_op_ratio = (usage["EQ"] / eq_plus_range) if eq_plus_range > 0 else 0.0

        row_count_log = math.log1p(col.row_count) 

        avg_latency_norm = self._normalize_latency(avg_latency, latency_stats)

        return [
            is_integer, is_text, is_real, is_blob,
            1.0 if col.is_pk else 0.0,
            1.0 if col.is_fk else 0.0,
            1.0 if col.is_indexed else 0.0,
            col.null_ratio,
            col.cardinality_ratio,
            row_count_log,
            where_freq, join_freq, orderby_freq, groupby_freq,
            equality_op_ratio,
            avg_latency_norm,
        ]

    @staticmethod
    def _normalize_latency(value: float, latency_stats: Dict) -> float:
        """Min-max normalize against the max latency seen across all columns in this DB."""
        if not latency_stats:
            return 0.0
        max_latency = max(latency_stats.values(), default=0.0)
        if max_latency <= 0:
            return 0.0
        return min(value / max_latency, 1.0)


if __name__ == "__main__":
    import sqlite3
    import os
    import tempfile
    from logger import LoggedConnection

    # build a small demo DB + generate some query traffic + logs
    db_path = os.path.join(tempfile.gettempdir(), "features_demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    setup_conn = sqlite3.connect(db_path)
    setup_conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    setup_conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)")
    setup_conn.executemany("INSERT INTO users (name, email) VALUES (?, ?)",
                            [(f"user{i}", f"user{i}@x.com") for i in range(50)])
    setup_conn.executemany("INSERT INTO orders (user_id, amount, status) VALUES (?, ?, ?)",
                            [(i % 50, 10.0 * i, "shipped" if i % 2 == 0 else "pending") for i in range(200)])
    setup_conn.commit()
    setup_conn.close()

    log_path = os.path.join(tempfile.gettempdir(), "features_demo.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    conn = LoggedConnection(db_path, log_path=log_path)
    for i in range(30):
        conn.execute("SELECT * FROM orders WHERE user_id = ?", (i % 50,))
    for i in range(15):
        conn.execute("SELECT * FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = ?", ("shipped",))
    conn.execute("SELECT * FROM orders ORDER BY amount")
    conn.close()

    extractor = FeatureExtractor(db_path, log_path)
    state = extractor.build_state()

    print(f"Feature order: {FEATURE_NAMES}\n")
    for row in state:
        print(f"{row.table}.{row.column}: {[round(v, 3) for v in row.vector]}")