"""Local demo authentication: users stored in SQLite with salted PBKDF2-SHA256 hashes.

This is a prototype login for the demonstration only - there is no password reset,
account management UI or transport security beyond what Streamlit provides locally.
The users table is deliberately excluded from "Reset demo data" and CSV export.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from . import db

ITERATIONS = 200_000

ROLES = {
    "Administrator": "Full access, including resetting the demo dataset.",
    "Maintenance Planner": "Full operational access; cannot reset the demo dataset.",
}

# Demo accounts created on first launch (shown on the login page - fictional demo only).
DEMO_USERS = [
    ("admin", "admin@123", "Demo Administrator", "Administrator"),
    ("planner", "planner@123", "Demo Maintenance Planner", "Maintenance Planner"),
]

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL,
    salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login TEXT
);
"""


def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex()


def create_user(username: str, password: str, full_name: str, role: str) -> None:
    username = username.strip().lower()
    if not username or not password:
        raise ValueError("Username and password are required.")
    if role not in ROLES:
        raise ValueError(f"Role must be one of {list(ROLES)}.")
    salt = secrets.token_bytes(16)
    with db.connect() as conn:
        conn.execute(USERS_SCHEMA)
        conn.execute("INSERT INTO users VALUES (?,?,?,?,?,?,NULL)",
                     (username, full_name, role, salt.hex(), hash_password(password, salt), db.now_iso()))


def ensure_users() -> None:
    """Create the users table and demo accounts if they do not exist yet."""
    with db.connect() as conn:
        conn.execute(USERS_SCHEMA)
        existing = {r[0] for r in conn.execute("SELECT username FROM users")}
    for username, password, name, role in DEMO_USERS:
        if username not in existing:
            create_user(username, password, name, role)


def authenticate(username: str, password: str) -> dict | None:
    """Return the user (without secrets) if the credentials are valid, else None."""
    username = (username or "").strip().lower()
    row = db.query_one("SELECT * FROM users WHERE username = ?", (username,))
    if row is None:
        hash_password(password or "", b"timing-equaliser")  # similar cost for unknown users
        return None
    if not hmac.compare_digest(hash_password(password or "", bytes.fromhex(row["salt"])), row["password_hash"]):
        return None
    with db.connect() as conn:
        conn.execute("UPDATE users SET last_login = ? WHERE username = ?", (db.now_iso(), username))
        db.log_activity(conn, "Access", f"{username} signed in")
    return {"username": row["username"], "full_name": row["full_name"], "role": row["role"]}


def can_reset(user: dict | None) -> bool:
    return bool(user) and user.get("role") == "Administrator"
