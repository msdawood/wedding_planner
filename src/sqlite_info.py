import sqlite3

conn = sqlite3.connect("resources/Chinook.db")
cur = conn.cursor()

# List all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = [row[0] for row in cur.fetchall()]
print("Tables:", tables)

# Print the schema (CREATE statements) for each table, index, view
cur.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL;")
for obj_type, name, sql in cur.fetchall():
    print(f"\n-- {obj_type}: {name}\n{sql}")

# Column details per table (name, type, not-null, default, primary key)
for t in tables:
    print(f"\n{t}:")
    for col in cur.execute(f"PRAGMA table_info('{t}');"):
        print("  ", col)

conn.close()