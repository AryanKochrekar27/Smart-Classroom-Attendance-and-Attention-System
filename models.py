# models.py

import os
import re
import sqlite3
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

DB_BACKEND = os.getenv("DB_BACKEND", "auto").strip().lower()  # auto | mysql | sqlite
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join("data", "app_attendance.db"))

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", "aiml"),
    "database": os.getenv("DB_NAME", "attendance_db"),
    "autocommit": False,
    "connection_timeout": 2,
}

_active_backend = None
_sqlite_initialized = False

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS semesters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    semester TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    semester TEXT NOT NULL,
    UNIQUE (name, semester)
);

CREATE TABLE IF NOT EXISTS timetable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day_of_week TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS teachers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    usn_start TEXT NOT NULL,
    usn_end TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    usn TEXT UNIQUE,
    semester TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL UNIQUE,
    vector_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER,
    subject_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    batch_id INTEGER NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE SET NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
    FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('Present', 'Absent')),
    time_marked TEXT NULL,
    attention_score REAL NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, subject_id, student_id),
    FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_students_semester ON students (semester);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance (date);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions (date);
"""


def _sqlite_field(value, *options):
    if value is None:
        return 0
    try:
        return options.index(value) + 1
    except ValueError:
        return 0


def _ensure_sqlite_initialized():
    global _sqlite_initialized
    if _sqlite_initialized:
        return
    db_dir = os.path.dirname(SQLITE_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SQLITE_SCHEMA)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(attendance)").fetchall()}
    if "attention_score" not in cols:
        conn.execute("ALTER TABLE attendance ADD COLUMN attention_score REAL NULL")
    conn.commit()
    conn.close()
    _sqlite_initialized = True


def _get_sqlite_connection():
    _ensure_sqlite_initialized()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.create_function("FIELD", -1, _sqlite_field)
    return conn


def _translate_mysql_to_sqlite(query):
    q = query.replace("%s", "?")
    q = re.sub(r"\bINSERT\s+IGNORE\s+INTO\b", "INSERT OR IGNORE INTO", q, flags=re.IGNORECASE)
    if re.search(r"\bON\s+DUPLICATE\s+KEY\s+UPDATE\b", q, flags=re.IGNORECASE):
        q = re.sub(
            r"\bON\s+DUPLICATE\s+KEY\s+UPDATE\b",
            "ON CONFLICT(date, subject_id, student_id) DO UPDATE SET",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(r"\bVALUES\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)", r"excluded.\1", q, flags=re.IGNORECASE)
    return q


def _resolve_backend():
    global _active_backend
    if _active_backend is not None:
        return _active_backend
    if DB_BACKEND == "sqlite":
        _ensure_sqlite_initialized()
        _active_backend = "sqlite"
        return _active_backend
    if DB_BACKEND == "mysql":
        _active_backend = "mysql"
        return _active_backend

    # auto mode: prefer MySQL if reachable, otherwise fall back to local SQLite
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        conn.close()
        _active_backend = "mysql"
    except Error as e:
        print(f"[DB] MySQL unavailable ({e}); using local SQLite database at {SQLITE_DB_PATH}")
        _ensure_sqlite_initialized()
        _active_backend = "sqlite"
    return _active_backend


def get_db_connection():
    backend = _resolve_backend()
    if backend == "sqlite":
        return _get_sqlite_connection()
    return mysql.connector.connect(**DB_CONFIG)


def fetch_all(query, params=()):
    backend = _resolve_backend()
    conn = get_db_connection()
    if backend == "sqlite":
        cursor = conn.cursor()
        cursor.execute(_translate_mysql_to_sqlite(query), params)
        rows = [dict(r) for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        return rows
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def fetch_one(query, params=()):
    backend = _resolve_backend()
    conn = get_db_connection()
    if backend == "sqlite":
        cursor = conn.cursor()
        cursor.execute(_translate_mysql_to_sqlite(query), params)
        row = cursor.fetchone()
        result = dict(row) if row is not None else None
        cursor.close()
        conn.close()
        return result
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row


def execute(query, params=()):
    backend = _resolve_backend()
    conn = get_db_connection()
    if backend == "sqlite":
        cursor = conn.cursor()
        cursor.execute(_translate_mysql_to_sqlite(query), params)
        conn.commit()
        lastid = cursor.lastrowid
        cursor.close()
        conn.close()
        return lastid
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    lastid = cursor.lastrowid
    cursor.close()
    conn.close()
    return lastid


def execute_many(query, param_list):
    backend = _resolve_backend()
    conn = get_db_connection()
    if backend == "sqlite":
        cursor = conn.cursor()
        cursor.executemany(_translate_mysql_to_sqlite(query), param_list)
        conn.commit()
        cursor.close()
        conn.close()
        return
    cursor = conn.cursor()
    cursor.executemany(query, param_list)
    conn.commit()
    cursor.close()
    conn.close()