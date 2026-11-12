import os
import cv2
import face_recognition
import pandas as pd
import mysql.connector
import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Database Connection ---
db = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASS", ""),
    database=os.getenv("DB_NAME", "attendance_db")
)
cursor = db.cursor(dictionary=True)

# --- Load Known Faces ---
df = pd.read_csv("data/features_all.csv")
known_encodings = df.iloc[:, 3:].values.tolist()
known_names = df["display_name"].tolist()
known_usn = df["usn"].tolist()
known_sem = df["semester"].tolist()

# --- Globals ---
attendance_marked = set()
camera_running = False
subject_selected = ""
semester_selected = ""
date_selected = datetime.date.today().strftime("%Y-%m-%d")


def set_session_data(subject, semester, date):
    """Set current session data from Flask form."""
    global subject_selected, semester_selected, date_selected, attendance_marked
    subject_selected = subject
    semester_selected = semester
    date_selected = date
    attendance_marked = set()


def gen_frames():
    """Generate frames from webcam and mark attendance."""
    global attendance_marked, camera_running
    cap = cv2.VideoCapture(0)
    camera_running = True

    while camera_running:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                idx = matches.index(True)
                name, usn, sem = known_names[idx], known_usn[idx], known_sem[idx]

                if sem == semester_selected and usn not in attendance_marked:
                    cursor.execute("SELECT id FROM students WHERE usn=%s", (usn,))
                    student = cursor.fetchone()
                    cursor.execute("SELECT id FROM subjects WHERE name=%s AND semester=%s",
                                   (subject_selected, sem))
                    subject = cursor.fetchone()

                    if student and subject:
                        cursor.execute("""
                            INSERT INTO attendance (date, subject_id, student_id, status, time_marked)
                            VALUES (%s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE status=VALUES(status)
                        """, (date_selected, subject["id"], student["id"], "Present",
                              datetime.datetime.now().strftime("%H:%M:%S")))
                        db.commit()
                        attendance_marked.add(usn)

                # Draw face box and label
                top, right, bottom, left = face_loc
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({usn})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Stream video frame to Flask
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def stop_camera_and_mark_absentees():
    """Stop the camera and mark absentees as 'Absent'."""
    global camera_running
    camera_running = False

    cursor.execute("SELECT usn FROM students WHERE semester=%s", (semester_selected,))
    all_usns = [row["usn"] for row in cursor.fetchall()]
    absentees = set(all_usns) - attendance_marked

    for usn in absentees:
        cursor.execute("SELECT id FROM students WHERE usn=%s", (usn,))
        student = cursor.fetchone()
        cursor.execute("SELECT id FROM subjects WHERE name=%s AND semester=%s",
                       (subject_selected, semester_selected))
        subject = cursor.fetchone()

        if student and subject:
            cursor.execute("""
                INSERT INTO attendance (date, subject_id, student_id, status)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE status=VALUES(status)
            """, (date_selected, subject["id"], student["id"], "Absent"))
    db.commit()
