import sqlite3
import pytest
from features import FeatureExtractor
from logger import LoggedConnection

@pytest.fixture
def features_db_log(tmp_path):
    db_path = str(tmp_path / 'feat.db')
    log_path = str(tmp_path / 'feat.jsonl')
    setup_conn = sqlite3.connect(db_path)
    setup_conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)')
    setup_conn.executemany('INSERT INTO t (val) VALUES (?)', [('A',), ('A',), ('B',), ('C',)])
    setup_conn.commit()
    setup_conn.close()
    
    conn = LoggedConnection(db_path, log_path=log_path)
    for _ in range(5): conn.execute('SELECT * FROM t WHERE val = "A"')
    conn.close()
    return db_path, log_path

def test_features(features_db_log):
    db_path, log_path = features_db_log
    extractor = FeatureExtractor(db_path, log_path)
    state = extractor.build_state()
    val_row = next(r for r in state if r.column == 'val')
    assert abs(val_row.vector[8] - 0.75) < 1e-5
    assert abs(val_row.vector[10] - 1.0) < 1e-5
