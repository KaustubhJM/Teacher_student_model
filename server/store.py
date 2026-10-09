"""Class records for the website, one JSON document per class in SQLite
(server/data/classes.db). Separate from the CLI's classroom_*.db files.

A record holds everything the website needs to show and resume a class:
title, sources, status, outline, per-section context chunks, generated
beats, transcript, doubt history, quiz and quiz score. Fields starting
with "_" are server-only (file paths, context, transcript) and are never
sent to the browser."""
import json
import os
import sqlite3
import threading
from typing import List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UPLOAD_DIR = os.environ.get("CLASSROOM_WEB_UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))
DB_PATH = os.environ.get("CLASSROOM_WEB_DB_PATH", os.path.join(DATA_DIR, "classes.db"))

_lock = threading.RLock()   # re-entrant: update() holds it across get + save


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS classes (id TEXT PRIMARY KEY, created REAL, data TEXT)")
    return conn


def save(cls: dict) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO classes (id, created, data) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
            (cls["id"], cls.get("_created", 0), json.dumps(cls)),
        )


def get(class_id: str) -> Optional[dict]:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT data FROM classes WHERE id = ?", (class_id,)).fetchone()
    return json.loads(row[0]) if row else None


def list_all() -> List[dict]:
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT data FROM classes ORDER BY created DESC").fetchall()
    return [json.loads(r[0]) for r in rows]


def delete(class_id: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM classes WHERE id = ?", (class_id,))


def update(class_id: str, fn=None, **fields) -> Optional[dict]:
    """Atomic read-modify-write of one record: merges `fields`, then calls
    fn(cls) (to append to lists etc.) if given. None if the class is gone."""
    with _lock:
        cls = get(class_id)
        if cls is None:
            return None
        cls.update(fields)
        if fn is not None:
            fn(cls)
        save(cls)
        return cls


def public(cls: dict) -> dict:
    """The record as the browser sees it (no server-only fields)."""
    return {k: v for k, v in cls.items() if not k.startswith("_")}
