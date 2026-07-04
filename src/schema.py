import sqlite3
import contextlib
from dataclasses import dataclass
from typing import List

@dataclass
class ColumnInfo:
    table_name: str
    column_name: str
    dtype: str
    is_pk: bool
    is_fk: bool
    is_indexed: bool
    null_ratio: float
    cardinality_ratio: float
    row_count: int

class SchemaReader:
    """
    Extracts the schema and basic statistics of a SQLite database across all tables.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_all_columns(self) -> List[ColumnInfo]:
        """
        Returns a flat list of ColumnInfo objects — one per column, across every table.
        """
        columns = []
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all user tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row['name'] for row in cursor.fetchall()]
            
            for table in tables:
                # 1. Row count
                cursor.execute(f'SELECT count(*) as cnt FROM "{table}"')
                row_count = cursor.fetchone()['cnt']
                
                # 2. Table info (columns, dtype, pk)
                cursor.execute(f'PRAGMA table_info("{table}")')
                table_info = cursor.fetchall()
                
                # 3. Foreign keys
                cursor.execute(f'PRAGMA foreign_key_list("{table}")')
                fk_list = cursor.fetchall()
                fk_columns = {row['from'] for row in fk_list}
                
                # 4. Indices
                cursor.execute(f'PRAGMA index_list("{table}")')
                index_list = cursor.fetchall()
                indexed_columns = set()
                for idx in index_list:
                    cursor.execute(f'PRAGMA index_info("{idx["name"]}")')
                    idx_info = cursor.fetchall()
                    for col in idx_info:
                        if col['name'] is not None:
                            indexed_columns.add(col['name'])
                            
                # 5. Process each column
                for col_info in table_info:
                    col_name = col_info['name']
                    dtype = col_info['type'].upper() if col_info['type'] else "UNKNOWN"
                    is_pk = bool(col_info['pk'])
                    is_fk = col_name in fk_columns
                    is_indexed = is_pk or (col_name in indexed_columns)
                    
                    # Compute null ratio and cardinality ratio
                    if row_count > 0:
                        cursor.execute(f'SELECT count("{col_name}") as non_null_cnt, count(DISTINCT "{col_name}") as distinct_cnt FROM "{table}"')
                        stats = cursor.fetchone()
                        non_null_cnt = stats['non_null_cnt']
                        distinct_cnt = stats['distinct_cnt']
                        
                        null_ratio = 1.0 - (non_null_cnt / row_count)
                        cardinality_ratio = distinct_cnt / non_null_cnt if non_null_cnt > 0 else 0.0
                    else:
                        null_ratio = 0.0
                        cardinality_ratio = 0.0
                        
                    columns.append(ColumnInfo(
                        table_name=table,
                        column_name=col_name,
                        dtype=dtype,
                        is_pk=is_pk,
                        is_fk=is_fk,
                        is_indexed=is_indexed,
                        null_ratio=null_ratio,
                        cardinality_ratio=cardinality_ratio,
                        row_count=row_count
                    ))
                    
        return columns

if __name__ == "__main__":
    import os
    # Quick smoke test
    db = "test_schema.db"
    # Ensure a clean slate
    if os.path.exists(db):
        os.remove(db)
        
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, group_id INTEGER, age INTEGER)")
        conn.execute("CREATE INDEX idx_group ON users(group_id)")
        conn.execute("INSERT INTO users (name, group_id, age) VALUES ('Alice', 1, 30)")
        conn.execute("INSERT INTO users (name, group_id, age) VALUES ('Bob', 1, NULL)")
        conn.execute("INSERT INTO users (name, group_id, age) VALUES ('Charlie', 2, 25)")
        conn.commit()

    reader = SchemaReader(db)
    cols = reader.get_all_columns()
    for c in cols:
        print(c)
        
    # Cleanup
    if os.path.exists(db):
        try:
            os.remove(db)
        except PermissionError:
            pass
