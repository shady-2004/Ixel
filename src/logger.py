import sqlite3
import time
import json
import threading
from pathlib import Path


class LoggedConnection:
    def __init__(self, db_path: str, log_path: str = None):
        self.db_path = db_path

        if log_path is None:
            db_name = Path(db_path).stem if db_path != ":memory:" else "memory"
            log_path = f"logs/{db_name}.jsonl"

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

        self._lock = threading.Lock()


    def _write_log(self, sql: str, params, latency_ms: float, error: str = None):
        entry = {
            "timestamp": time.time(),
            "db_path": self.db_path,
            "sql": sql.strip(),
            "params": self._safe_params(params),
            "latency_ms": round(latency_ms, 4),
            "error": error,
        }
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _safe_params(params):
        
        if params is None:
            return None
        try:
            return [p if isinstance(p, (int, float, str, bool)) else str(type(p)) for p in params]
        except TypeError:
            return str(params)

    # ---- wrapped execution methods ----

    def execute(self, sql: str, params=()):
        start = time.perf_counter()
        error = None
        try:
            cur = self._conn.execute(sql, params)
        except sqlite3.Error as e:
            error = str(e)
            latency_ms = (time.perf_counter() - start) * 1000
            self._write_log(sql, params, latency_ms, error)
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        self._write_log(sql, params, latency_ms)
        return cur

    def executemany(self, sql: str, seq_of_params):
        start = time.perf_counter()
        error = None
        try:
            cur = self._conn.executemany(sql, seq_of_params)
        except sqlite3.Error as e:
            error = str(e)
            latency_ms = (time.perf_counter() - start) * 1000
            self._write_log(sql, f"<{len(list(seq_of_params))} param sets>", latency_ms, error)
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        self._write_log(sql, f"<batch>", latency_ms)
        return cur

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


if __name__ == "__main__":
    # quick smoke test
    conn = LoggedConnection(":memory:", log_path="logs/demo.jsonl")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES (?)", ("Shady",))
    conn.execute("INSERT INTO users (name) VALUES (?)", ("Ahmed",))
    cur = conn.execute("SELECT * FROM users WHERE name = ?", ("Shady",))
    print(cur.fetchall())
    conn.commit()
    conn.close()
    print("Logged queries written to logs/demo.jsonl")
