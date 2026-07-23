import os
import sqlite3
import pytest
from schema import SchemaReader

@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / 'test.db'
    conn = sqlite3.connect(str(db_path))
    conn.execute('CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES b(id), val TEXT UNIQUE)')
    conn.execute('CREATE TABLE b (id INTEGER PRIMARY KEY, name TEXT NOT NULL)')
    conn.execute('CREATE INDEX idx_val ON a(val)')
    conn.commit()
    conn.close()
    return str(db_path)

def test_schema_parsing(test_db):
    reader = SchemaReader(test_db)
    cols = {c.column_name: c for c in reader.get_all_columns()}
    assert cols['id'].is_pk is True
    assert cols['b_id'].is_fk is True
    assert cols['val'].is_indexed is True
