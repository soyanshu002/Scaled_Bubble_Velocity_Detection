import sqlite3
import os

# Detect project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "output", "bubble_info.db")

# Make sure folder exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS zone_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            run_name TEXT,
            zone_name TEXT,
            avg_small_count REAL,
            avg_medium_count REAL,
            avg_large_count REAL,
            avg_small_velocity REAL,
            avg_medium_velocity REAL,
            avg_large_velocity REAL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    conn.commit()
    conn.close()


def insert_run(run_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("INSERT OR IGNORE INTO runs (run_name) VALUES (?)", (run_name,))
    conn.commit()

    cursor.execute("SELECT id FROM runs WHERE run_name = ?", (run_name,))
    run_id = cursor.fetchone()[0]

    conn.close()
    return run_id


def insert_zone_metrics(run_id, run_name, zone_name,
                        avg_small_count, avg_medium_count, avg_large_count,
                        avg_small_velocity, avg_medium_velocity, avg_large_velocity):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO zone_metrics (
            run_id, run_name, zone_name,
            avg_small_count, avg_medium_count, avg_large_count,
            avg_small_velocity, avg_medium_velocity, avg_large_velocity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, run_name, zone_name,
        avg_small_count, avg_medium_count, avg_large_count,
        avg_small_velocity, avg_medium_velocity, avg_large_velocity
    ))

    conn.commit()
    conn.close()
