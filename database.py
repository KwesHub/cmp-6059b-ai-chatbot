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

    conn.commit()
    conn.close()
    print("Database initialised successfully.")


def seed_station_codes():
    """Insert the core station codes if not already present."""
    stations = [
        ("Norwich", "NRW"),
        ("London Liverpool Street", "LST"),
        ("London Waterloo", "WAT"),
        ("London Paddington", "PAD"),
        ("Oxford", "OXF"),
        ("Weymouth", "WEY"),
        ("Southampton Central", "SOU"),
        ("Winchester", "WIN"),
        ("Bournemouth", "BMH"),
        ("Poole", "POO"),
        ("Wareham", "WRM"),
        ("Dorchester South", "DCH"),
        ("Cambridge", "CBG"),
        ("Birmingham New Street", "BHM"),
        ("Manchester Piccadilly", "MAN"),
        ("Bristol Temple Meads", "BRI"),
        ("Edinburgh Waverley", "EDB"),
        ("Glasgow Central", "GLC"),
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