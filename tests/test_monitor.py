import sqlite3
from monitor import Monitor
from logger import LoggedConnection

def test_monitor(tmp_path):
    db_path = str(tmp_path / 'mon.db')
    log_path = str(tmp_path / 'mon.jsonl')
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)')
    conn.execute('CREATE INDEX idx_a ON t(a)')
    conn.commit()
    conn.close()
    
    lc = LoggedConnection(db_path, log_path=log_path)
    for _ in range(3): lc.execute('SELECT * FROM t WHERE a=1')
    for _ in range(7): lc.execute('SELECT * FROM t')
    lc.close()
    
    mon = Monitor(db_path, log_path, window_seconds=99999, stale_hit_rate_threshold=0.01, workload_shift_threshold=0.2)
    rep1 = mon.check()
    assert len(rep1.stale_indexes) == 0
    
    lc = LoggedConnection(db_path, log_path=log_path)
    for _ in range(990): lc.execute('SELECT * FROM t')
    lc.close()
    
    rep2 = mon.check()
    assert 'idx_a' in rep2.stale_indexes
    assert rep2.workload_shifted is True
