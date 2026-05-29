import sqlite3
import threading
from datetime import datetime

# Thread-safe connection per thread
_local = threading.local()

DB_PATH = "/tmp/chat.db"


def get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect("/tmp/chat.db", check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            title       TEXT DEFAULT 'New Chat',
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            message     TEXT NOT NULL,
            timestamp   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            summary     TEXT NOT NULL,
            up_to_id    INTEGER NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_history_session ON chat_history(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
    """)
    conn.commit()


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(session_id: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
        (session_id,)
    )
    conn.commit()


def set_title(session_id: str, title: str):
    conn = get_conn()
    conn.execute(
        "UPDATE sessions SET title=?, updated_at=datetime('now') WHERE session_id=?",
        (title[:80], session_id)
    )
    conn.commit()


def get_sessions(limit: int = 40):
    conn = get_conn()
    rows = conn.execute(
        "SELECT session_id, title, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()


# ── Messages ──────────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, message: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?)",
        (session_id, role, message)
    )
    conn.execute(
        "UPDATE sessions SET updated_at=datetime('now') WHERE session_id=?",
        (session_id,)
    )
    conn.commit()
    return cur.lastrowid


def get_history(session_id: str, limit: int = 60):
    """Return last `limit` messages as list of dicts."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, role, message, timestamp FROM chat_history
           WHERE session_id=? ORDER BY id DESC LIMIT ?""",
        (session_id, limit)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_message_count(session_id: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as c FROM chat_history WHERE session_id=?",
        (session_id,)
    ).fetchone()
    return row["c"] if row else 0


# ── Summaries ─────────────────────────────────────────────────────────────────

def save_summary(session_id: str, summary: str, up_to_id: int):
    conn = get_conn()
    conn.execute(
        "INSERT INTO summaries (session_id, summary, up_to_id) VALUES (?, ?, ?)",
        (session_id, summary, up_to_id)
    )
    conn.commit()


def get_latest_summary(session_id: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT summary, up_to_id FROM summaries WHERE session_id=? ORDER BY id DESC LIMIT 1",
        (session_id,)
    ).fetchone()
    return dict(row) if row else None


def get_messages_after(session_id: str, after_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, role, message FROM chat_history WHERE session_id=? AND id > ? ORDER BY id ASC",
        (session_id, after_id)
    ).fetchall()
    return [dict(r) for r in rows]


init_db()