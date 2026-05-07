import sqlite3
from config import DB_PATH


def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def initialise_database():
    """Create all tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ─── Conversation history ────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            user_msg  TEXT    NOT NULL,
            bot_msg   TEXT    NOT NULL,
            intent    TEXT
        )
    """)

    # ─── Station codes ───────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS station_codes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            station_name TEXT NOT NULL UNIQUE,
            crs_code     TEXT NOT NULL
        )
    """)

    # ─── Historical train data (Task 2) ──────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS train_performance (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            service_date        TEXT,
            origin              TEXT,
            destination         TEXT,
            scheduled_departure TEXT,
            actual_departure    TEXT,
            scheduled_arrival   TEXT,
            actual_arrival      TEXT,
            delay_minutes       REAL
        )
    """)

    # ─── User sessions ───────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            started_at TEXT NOT NULL,
            intent     TEXT,
            origin     TEXT,
            destination TEXT,
            travel_date TEXT,
            completed  INTEGER DEFAULT 0
        )
    """)

    # ─── Knowledge Base Q&A ─────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kb_qa (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            category    TEXT NOT NULL,
            question    TEXT NOT NULL,
            keywords    TEXT,  -- JSON list
            answer      TEXT NOT NULL
        )
    """)

    # ─── Knowledge Base Rules ───────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kb_rules (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_name   TEXT NOT NULL UNIQUE,
            rule_data   TEXT NOT NULL  -- JSON
        )
    """)

    # ─── Knowledge Base Fallbacks ───────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kb_fallbacks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            response    TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialised successfully.")


def seed_station_codes():
    """
    Load ALL stations from data/StationNameAndCode.csv into the DB.
    Falls back to a hardcoded list if the CSV isn't found.
    """
    import os, csv
    csv_path = os.path.join(os.path.dirname(__file__), "data", "StationNameAndCode.csv")

    stations = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    name = row[0].strip().title()   # "SHEFFIELD" → "Sheffield"
                    crs  = row[1].strip().upper()
                    if name and crs:
                        stations.append((name, crs))
        print(f"Loading {len(stations)} stations from CSV...")
    else:
        # Fallback hardcoded list
        stations = [
            ("Norwich", "NRW"), ("London Liverpool Street", "LST"),
            ("London Waterloo", "WAT"), ("London Paddington", "PAD"),
            ("Oxford", "OXF"), ("Weymouth", "WEY"),
            ("Southampton Central", "SOU"), ("Winchester", "WIN"),
            ("Bournemouth", "BMH"), ("Poole", "POO"),
            ("Wareham", "WRM"), ("Dorchester South", "DCH"),
            ("Cambridge", "CBG"), ("Birmingham New Street", "BHM"),
            ("Manchester Piccadilly", "MAN"), ("Bristol Temple Meads", "BRI"),
            ("Edinburgh Waverley", "EDB"), ("Glasgow Central", "GLC"),
            ("Sheffield", "SHF"), ("Leeds", "LDS"),
            ("Liverpool Lime Street", "LIV"), ("Brighton", "BTN"),
            ("Exeter St Davids", "EXD"), ("Cardiff Central", "CDF"),
        ]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO station_codes (station_name, crs_code)
        VALUES (?, ?)
    """, stations)
    conn.commit()
    conn.close()
    print(f"Seeded {len(stations)} station codes.")


if __name__ == "__main__":
    initialise_database()
    seed_station_codes()