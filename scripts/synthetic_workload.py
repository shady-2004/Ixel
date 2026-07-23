import os
import random
import string
import sqlite3
from pathlib import Path
from logger import LoggedConnection

DATA_DIR = Path("./data")
LOGS_DIR = DATA_DIR / "logs"

def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_db_1():
    """Small DB: Users and simple lookups."""
    db_path = DATA_DIR / "training_db_1.db"
    log_path = LOGS_DIR / "training_db_1.jsonl"
    if db_path.exists(): db_path.unlink()
    if log_path.exists(): log_path.unlink()
    
    conn = LoggedConnection(str(db_path), log_path=str(log_path))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, status TEXT, age INTEGER)")
    
    users = [(random_string(8), random.choice(['active', 'inactive', 'banned']), random.randint(18, 80)) for _ in range(1000)]
    conn.executemany("INSERT INTO users (username, status, age) VALUES (?, ?, ?)", users)
    conn.commit()
    
    # Workload
    for _ in range(500):
        # Point lookups on username
        username = random.choice(users)[0]
        conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        
        # Scans on status
        status = random.choice(['active', 'inactive'])
        conn.execute("SELECT id FROM users WHERE status = ?", (status,))
        
        # Range scans on age
        conn.execute("SELECT count(*) FROM users WHERE age > 30")
        
    conn.close()
    print("Created training_db_1")

def create_db_2():
    """Medium DB: E-commerce style, multiple tables, joins."""
    db_path = DATA_DIR / "training_db_2.db"
    log_path = LOGS_DIR / "training_db_2.jsonl"
    if db_path.exists(): db_path.unlink()
    if log_path.exists(): log_path.unlink()
    
    conn = LoggedConnection(str(db_path), log_path=str(log_path))
    conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price REAL)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER, user_id INTEGER, amount REAL)")
    
    products = [(random.choice(['Electronics', 'Books', 'Clothing']), round(random.uniform(10, 500), 2)) for _ in range(500)]
    conn.executemany("INSERT INTO products (category, price) VALUES (?, ?)", products)
    
    orders = [(random.randint(1, 500), random.randint(1, 1000), round(random.uniform(10, 500), 2)) for _ in range(5000)]
    conn.executemany("INSERT INTO orders (product_id, user_id, amount) VALUES (?, ?, ?)", orders)
    conn.commit()
    
    # Workload
    for _ in range(800):
        # Join query
        user_id = random.randint(1, 1000)
        conn.execute("SELECT p.category, o.amount FROM orders o JOIN products p ON o.product_id = p.id WHERE o.user_id = ?", (user_id,))
        
        # Aggregation
        cat = random.choice(['Electronics', 'Books', 'Clothing'])
        conn.execute("SELECT count(*), sum(price) FROM products WHERE category = ?", (cat,))
        
        # Ordering
        conn.execute("SELECT * FROM orders ORDER BY amount DESC LIMIT 10")
        
    conn.close()
    print("Created training_db_2")

def create_db_3():
    """Heavy writes DB: Logging system."""
    db_path = DATA_DIR / "training_db_3.db"
    log_path = LOGS_DIR / "training_db_3.jsonl"
    if db_path.exists(): db_path.unlink()
    if log_path.exists(): log_path.unlink()
    
    conn = LoggedConnection(str(db_path), log_path=str(log_path))
    conn.execute("CREATE TABLE server_logs (id INTEGER PRIMARY KEY, server_id TEXT, level TEXT, message TEXT, timestamp INTEGER)")
    
    logs = [(f"server_{random.randint(1, 10)}", random.choice(['INFO', 'WARN', 'ERROR']), random_string(20), i) for i in range(2000)]
    conn.executemany("INSERT INTO server_logs (server_id, level, message, timestamp) VALUES (?, ?, ?, ?)", logs)
    conn.commit()
    
    # Workload
    for i in range(2000, 2500):
        # Heavy writes
        conn.execute("INSERT INTO server_logs (server_id, level, message, timestamp) VALUES (?, ?, ?, ?)", 
                     (f"server_{random.randint(1, 10)}", random.choice(['INFO', 'WARN', 'ERROR']), random_string(20), i))
        
        if i % 5 == 0:
            # Occasional lookups
            conn.execute("SELECT * FROM server_logs WHERE level = 'ERROR' ORDER BY timestamp DESC LIMIT 50")
            
        if i % 10 == 0:
            conn.execute("SELECT count(*) FROM server_logs WHERE server_id = ?", (f"server_{random.randint(1, 10)}",))
            
    conn.commit()
    conn.close()
    print("Created training_db_3")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    LOGS_DIR = DATA_DIR / "logs"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    create_db_1()
    create_db_2()
    create_db_3()
    print("Synthetic workload generation complete.")
