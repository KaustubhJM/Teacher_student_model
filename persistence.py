"""
persistence.py — cross-run session tracking for the classroom.

LangGraph's checkpointer (SqliteSaver, wired in classroom.py) already
gives crash-safe, cross-process resume: as long as the SAME thread_id
is used, graph.invoke(None, config=config) picks up exactly where a
previous run left off, even after a full restart days later. That's
the "resume this exact lecture" half of persistence.

This module is the other half: a small SQLite table, separate from the
checkpointer's own storage, that answers questions the checkpointer
can't — "what has this student already studied", "is the last one
still in progress", "when did they last touch it" — since a finished
or abandoned thread's checkpoint isn't something you can list or query
across threads.
"""

import hashlib
import os
import re
import sqlite3
import time
from typing import List, Optional

DB_PATH = os.environ.get("CLASSROOM_DB_PATH", "classroom_progress.db")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def make_thread_id(student_name: str, topic: str) -> str:
    """Deterministic thread id from (student, topic) so re-running the
    same student+topic always resumes the same LangGraph thread instead
    of starting a new one. Short hash suffix guards against two
    different topics slugifying to the same string."""
    base = f"{_slug(student_name)}__{_slug(topic)}"
    h = hashlib.md5(f"{student_name}::{topic}".encode()).hexdigest()[:8]
    return f"{base}-{h}"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            topic TEXT NOT NULL,
            document_path TEXT,
            status TEXT NOT NULL DEFAULT 'in_progress',  -- in_progress | completed
            quiz_note TEXT,        -- e.g. "8 questions generated" — no grading exists yet
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    return conn


def get_session(thread_id: str) -> Optional[dict]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT thread_id, student_name, topic, document_path, status, quiz_note, "
            "created_at, updated_at FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    keys = ["thread_id", "student_name", "topic", "document_path", "status",
            "quiz_note", "created_at", "updated_at"]
    return dict(zip(keys, row))


def start_or_resume_session(thread_id: str, student_name: str, topic: str,
                             document_path: Optional[str]) -> dict:
    """Registers a session row the first time we see this thread_id, or
    just returns the existing row — regardless of how long ago it was
    last touched, which is what makes 'continue where I left off
    yesterday' work rather than just within one process."""
    existing = get_session(thread_id)
    if existing is not None:
        return existing

    conn = _connect()
    now = time.time()
    try:
        conn.execute(
            "INSERT INTO sessions (thread_id, student_name, topic, document_path, "
            "status, created_at, updated_at) VALUES (?, ?, ?, ?, 'in_progress', ?, ?)",
            (thread_id, student_name, topic, document_path, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return get_session(thread_id)


def mark_completed(thread_id: str, quiz_note: Optional[str] = None) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE sessions SET status = 'completed', quiz_note = ?, updated_at = ? "
            "WHERE thread_id = ?",
            (quiz_note, time.time(), thread_id),
        )
        conn.commit()
    finally:
        conn.close()


def touch(thread_id: str) -> None:
    """Bump updated_at — called after each section, so 'last worked on'
    reflects real progress, not just session start."""
    conn = _connect()
    try:
        conn.execute("UPDATE sessions SET updated_at = ? WHERE thread_id = ?",
                      (time.time(), thread_id))
        conn.commit()
    finally:
        conn.close()


def list_history(student_name: str, limit: int = 20) -> List[dict]:
    """All past sessions for this student, most recently touched first."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT thread_id, topic, status, quiz_note, created_at, updated_at "
            "FROM sessions WHERE student_name = ? ORDER BY updated_at DESC LIMIT ?",
            (student_name, limit),
        ).fetchall()
    finally:
        conn.close()
    keys = ["thread_id", "topic", "status", "quiz_note", "created_at", "updated_at"]
    return [dict(zip(keys, r)) for r in rows]