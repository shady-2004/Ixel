import sqlite3
from pipeline import Pipeline
from logger import LoggedConnection
from policy import BanditPolicy

def test_pipeline_rollback(tmp_path):
    db_path = str(tmp_path / 'pipe.db')
    log_path = str(tmp_path / 'pipe.jsonl')
    
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)')
    conn.executemany('INSERT INTO t(a) VALUES (?)', [(i,) for i in range(100)])
    conn.commit()
    conn.close()
    
    lc = LoggedConnection(db_path, log_path=log_path)
    for _ in range(10): lc.execute('SELECT * FROM t WHERE a=1')
    lc.close()
    
    model_path = str(tmp_path / 'model.pth')
    BanditPolicy(input_dim=16).save(model_path)
    
    pipe = Pipeline(db_path, log_path, model_path, reward_threshold=-100.0)
    pipe.policy.select_action = lambda vec, explore: ('INDEX', None)
    pipe.run()
    
    conn = sqlite3.connect(db_path)
    assert len(conn.execute("PRAGMA index_list('t')").fetchall()) > 0
    conn.close()
    
    pipe.rollback(2)
    
    conn = sqlite3.connect(db_path)
    assert len(conn.execute("PRAGMA index_list('t')").fetchall()) == 0
    conn.close()
