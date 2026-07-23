import pytest
import sqlite3
from simulator import Simulator
from logger import LoggedConnection

@pytest.fixture
def sim_env(tmp_path):
    db_path = str(tmp_path / 'sim.db')
    log_path = str(tmp_path / 'sim.jsonl')
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, good INTEGER, bad INTEGER, val INTEGER)')
    conn.executemany('INSERT INTO t (good, bad, val) VALUES (?, ?, ?)', [(i, i % 2, i) for i in range(1000)])
    conn.execute('CREATE INDEX idx_val_real ON t(val)')
    conn.commit()
    conn.close()
    
    lc = LoggedConnection(db_path, log_path=log_path)
    for i in range(30): lc.execute('SELECT * FROM t WHERE good = ?', (i,))
    for i in range(3): lc.execute('SELECT * FROM t WHERE bad = ?', (i,))
    for i in range(20): lc.execute('SELECT * FROM t WHERE val = ?', (i,))
    lc.execute('SELECT * FROM t WHERE good = ? AND bad = ?', (1, 1))
    lc.close()
    return db_path, log_path

def test_simulator_reward(sim_env):
    db_path, log_path = sim_env
    sim = Simulator(db_path, log_path)
    res_good = sim.evaluate('t', 'good', 'INDEX')
    res_bad = sim.evaluate('t', 'bad', 'INDEX')
    
    assert res_good.reward > 0
    assert res_bad.reward > 0
    
    res_drop = sim.evaluate('t', 'val', 'DROP')
    assert res_drop.reward < 0

def test_simulator_evaluate_batch(sim_env):
    db_path, log_path = sim_env
    sim = Simulator(db_path, log_path)
    res = sim.evaluate_batch([('t', 'good', 'INDEX'), ('t', 'bad', 'INDEX')])
    assert res.reward > 0
