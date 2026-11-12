# File: setup_db.py (Run once to initialize the new structure)

import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Students table with class, name, and USN
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    usn TEXT NOT NULL UNIQUE,
    class TEXT NOT NULL
)
""")

# Attendance table linked to students
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    student_id INTEGER,
    date DATE,
    time TEXT,
    FOREIGN KEY(student_id) REFERENCES students(id),
    UNIQUE(student_id, date)
)
""")

conn.commit()
conn.close()
print("Database setup complete.")
