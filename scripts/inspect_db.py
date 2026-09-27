"""Inspect current database state."""
import sqlite3
conn = sqlite3.connect('data/database/passport_screening.db')
conn.row_factory = sqlite3.Row

print("=== TABLES ===")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(tables)

for tbl in tables:
    print(f"\n=== {tbl} schema ===")
    schema = conn.execute(f"PRAGMA table_info('{tbl}')").fetchall()
    for r in schema:
        print(dict(r))
    cnt = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
    print(f"Row count: {cnt}")
    row = conn.execute(f"SELECT * FROM {tbl} LIMIT 1").fetchone()
    if row:
        print(f"Sample: {dict(row)}")

conn.close()
