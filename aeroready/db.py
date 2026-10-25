"""SQLite persistence layer: schema, connections and small query helpers.

A new connection is opened per operation, which keeps things safe across
Streamlit's script-runner threads. The database path can be redirected
(e.g. for tests) with :func:`set_db_path`.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config

_db_path: Path = config.DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS aircraft (
    tail_no TEXT PRIMARY KEY,
    aircraft_type TEXT NOT NULL,
    base TEXT NOT NULL,
    squadron TEXT NOT NULL,
    year_built INTEGER NOT NULL,
    total_flight_hours REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS components (
    component_id TEXT PRIMARY KEY,
    tail_no TEXT NOT NULL REFERENCES aircraft(tail_no),
    component_type TEXT NOT NULL,
    position TEXT NOT NULL,
    serial_no TEXT NOT NULL,
    install_date TEXT NOT NULL,
    operating_hours REAL NOT NULL,
    rated_life_hours REAL NOT NULL,
    hours_since_maintenance REAL NOT NULL,
    age_years REAL NOT NULL,
    prior_faults INTEGER NOT NULL DEFAULT 0,
    maintenance_count INTEGER NOT NULL DEFAULT 0,
    temp_deviation REAL NOT NULL,
    vibration REAL NOT NULL,
    pressure_deviation REAL NOT NULL,
    last_reading_at TEXT
);
CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id TEXT NOT NULL REFERENCES components(component_id),
    recorded_at TEXT NOT NULL,
    operating_hours REAL NOT NULL,
    hours_since_maintenance REAL NOT NULL,
    temp_deviation REAL NOT NULL,
    vibration REAL NOT NULL,
    pressure_deviation REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'telemetry'
);
CREATE INDEX IF NOT EXISTS ix_readings_component ON sensor_readings(component_id, recorded_at);
CREATE TABLE IF NOT EXISTS maintenance_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id TEXT NOT NULL REFERENCES components(component_id),
    tail_no TEXT NOT NULL REFERENCES aircraft(tail_no),
    performed_on TEXT NOT NULL,
    record_type TEXT NOT NULL,
    description TEXT NOT NULL,
    hours_at REAL,
    team_id TEXT,
    work_order_id TEXT
);
CREATE TABLE IF NOT EXISTS faults (
    fault_id TEXT PRIMARY KEY,
    tail_no TEXT NOT NULL REFERENCES aircraft(tail_no),
    component_id TEXT REFERENCES components(component_id),
    reported_on TEXT NOT NULL,
    fault_code TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    systems TEXT NOT NULL,
    hours_per_day REAL NOT NULL,
    service_only INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS spare_parts (
    part_number TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    component_type TEXT NOT NULL,
    part_role TEXT NOT NULL,
    qty_on_hand INTEGER NOT NULL,
    qty_reserved INTEGER NOT NULL DEFAULT 0,
    reorder_threshold INTEGER NOT NULL,
    lead_time_days INTEGER NOT NULL,
    unit_cost REAL NOT NULL,
    location TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id TEXT PRIMARY KEY,
    part_number TEXT NOT NULL REFERENCES spare_parts(part_number),
    qty INTEGER NOT NULL,
    ordered_on TEXT NOT NULL,
    expected_on TEXT NOT NULL,
    status TEXT NOT NULL,
    received_on TEXT
);
CREATE TABLE IF NOT EXISTS work_orders (
    wo_id TEXT PRIMARY KEY,
    tail_no TEXT NOT NULL REFERENCES aircraft(tail_no),
    component_id TEXT NOT NULL REFERENCES components(component_id),
    title TEXT NOT NULL,
    action TEXT NOT NULL,
    severity TEXT NOT NULL,
    risk_score REAL,
    recommended_action TEXT,
    part_number TEXT REFERENCES spare_parts(part_number),
    part_qty INTEGER NOT NULL DEFAULT 0,
    parts_reserved INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    team_id TEXT REFERENCES teams(team_id),
    scheduled_start TEXT,
    scheduled_end TEXT,
    est_hours REAL NOT NULL,
    grounding INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS maintenance_schedule (
    task_id TEXT PRIMARY KEY,
    tail_no TEXT NOT NULL REFERENCES aircraft(tail_no),
    task TEXT NOT NULL,
    interval_fh REAL NOT NULL,
    last_done_fh REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS activity_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL,
    tail_no TEXT
);
"""

TABLES = [
    "aircraft", "components", "sensor_readings", "maintenance_records", "faults", "teams",
    "spare_parts", "purchase_orders", "work_orders", "maintenance_schedule", "activity_log",
]


def set_db_path(path) -> None:
    global _db_path
    _db_path = Path(path)


def get_db_path() -> Path:
    return _db_path


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


@contextmanager
def connect():
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def query_df(sql: str, params: tuple | dict = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def query_one(sql: str, params: tuple = ()) -> dict | None:
    with connect() as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


def execute(sql: str, params: tuple | dict = ()) -> None:
    with connect() as conn:
        conn.execute(sql, params)


def table(name: str) -> pd.DataFrame:
    if name not in TABLES:
        raise ValueError(f"Unknown table: {name}")
    return query_df(f"SELECT * FROM {name}")


def log_activity(conn: sqlite3.Connection, category: str, message: str, tail_no: str | None = None) -> None:
    conn.execute(
        "INSERT INTO activity_log (ts, category, message, tail_no) VALUES (?, ?, ?, ?)",
        (now_iso(), category, message, tail_no),
    )


def get_meta(key: str) -> str | None:
    try:
        row = query_one("SELECT value FROM meta WHERE key = ?", (key,))
    except sqlite3.OperationalError:
        return None
    return row["value"] if row else None


def is_seeded() -> bool:
    return get_meta("seeded_at") is not None


def next_id(conn: sqlite3.Connection, table_name: str, column: str, prefix: str, width: int = 4) -> str:
    rows = conn.execute(f"SELECT {column} FROM {table_name} WHERE {column} LIKE ?", (f"{prefix}%",)).fetchall()
    numbers = [int(r[0][len(prefix):]) for r in rows if r[0][len(prefix):].isdigit()]
    return f"{prefix}{(max(numbers) + 1 if numbers else 1):0{width}d}"


def drop_all() -> None:
    with connect() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        for name in TABLES + ["meta"]:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
