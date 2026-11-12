# app.py
"""
Integrated Flask app for face-based attendance.
Features:
 - Admin: add semesters/subjects and a simple timetable
 - Register (upload) and Register-live (launch Tkinter registration GUI)
 - Process features (run features_extraction_to_csv.py) and refresh embeddings
 - Teacher: start session, take attendance by upload OR by server webcam capture
 - View attendance per-subject-per-day
"""
import os
import subprocess
import uuid
from datetime import datetime, date
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, abort, flash
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# local helpers (must exist)
from models import fetch_all, fetch_one, execute
import face_utils

load_dotenv()

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "data_faces")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB per upload
app.secret_key = os.getenv("SECRET_KEY", "change-me")

# In-memory embeddings cache: {student_id: vector_text}
embeddings_cache = {}

def refresh_embeddings_cache():
    global embeddings_cache
    try:
        rows = fetch_all("SELECT student_id, vector_text FROM embeddings")
        embeddings_cache = {r["student_id"]: r["vector_text"] for r in rows if r["vector_text"]}
        print("Embeddings cache refreshed, count:", len(embeddings_cache))
    except Exception as e:
        print("Failed refreshing embeddings cache:", e)

# initial load
try:
    refresh_embeddings_cache()
except Exception as e:
    print("Warning: could not load embeddings:", e)


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS

# ------ Home / Admin / Timetable routes ------

@app.route("/")
def index():
    semesters = fetch_all("SELECT DISTINCT semester FROM subjects ORDER BY semester")
    return render_template("index.html", semesters=semesters)

@app.route("/admin", methods=["GET", "POST"])
def page_admin():
    # Handle POST requests (both HTML form and JSON AJAX)
    if request.method == "POST":
        # ✅ Handle AJAX JSON requests first (edit/delete)
        if request.is_json:
            data = request.get_json()
            action = data.get("action")

            # --- Edit Subject ---
            if action == "edit_subject":
                sid = data.get("id")
                name = data.get("name", "").strip()
                sem = data.get("semester", "").strip()
                if not sid or not name or not sem:
                    return jsonify({"error": "id, name and semester are required"}), 400
                try:
                    execute("UPDATE subjects SET name=%s, semester=%s WHERE id=%s", (name, sem, sid))
                    return jsonify({"ok": True})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500

            # --- Delete Subject ---
            elif action == "delete_subject":
                sid = data.get("id")
                if not sid:
                    return jsonify({"error": "id required"}), 400
                try:
                    # Delete related timetable entries (optional cleanup)
                    execute("DELETE FROM timetable WHERE subject_id=%s", (sid,))
                    # Delete the subject itself
                    execute("DELETE FROM subjects WHERE id=%s", (sid,))
                    return jsonify({"ok": True})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500

        # ✅ Handle normal form submissions (add semester, subject, timetable)
        typ = request.form.get("type")

        # --- Add Semester ---
        if typ == "semester":
            sem = request.form.get("semester", "").strip()
            if sem:
                execute("INSERT IGNORE INTO semesters (semester) VALUES (%s)", (sem,))
                flash("Semester added")

        # --- Add Subject ---
        elif typ == "subject":
            name = request.form.get("name", "").strip()
            sem = request.form.get("semester", "").strip()
            if name and sem:
                execute("INSERT IGNORE INTO subjects (name, semester) VALUES (%s, %s)", (name, sem))
                flash("Subject added")

        # --- Add Timetable Entry ---
        elif typ == "timetable":
            day = request.form.get("day")
            subject_id = request.form.get("subject_id")
            start_time = request.form.get("start_time")
            end_time = request.form.get("end_time")
            if day and subject_id and start_time and end_time:
                execute(
                    "INSERT INTO timetable (day_of_week, subject_id, start_time, end_time) VALUES (%s, %s, %s, %s)",
                    (day, subject_id, start_time, end_time),
                )
                flash("Timetable entry added")

        # After any POST, refresh page
        return redirect(url_for("page_admin"))

    # ✅ GET: Load data for page display
    semesters = fetch_all("SELECT * FROM semesters ORDER BY semester")
    subjects = fetch_all("SELECT * FROM subjects ORDER BY semester, name")
    timetable = fetch_all(
        """
        SELECT t.id, t.day_of_week, t.start_time, t.end_time, s.name, s.semester
        FROM timetable t
        JOIN subjects s ON t.subject_id = s.id
        ORDER BY FIELD(t.day_of_week,
            'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'), t.start_time
        """
    )

    # Fetch batches if you are using the inline batch manager (optional)
    batches = fetch_all("""
        SELECT b.id, b.name, b.usn_start, b.usn_end, b.subject_id, s.name AS subject_name, s.semester
        FROM batches b
        JOIN subjects s ON s.id = b.subject_id
        ORDER BY s.semester, s.name, b.name
    """)

    return render_template(
        "admin.html",
        semesters=semesters,
        subjects=subjects,
        timetable=timetable,
        batches=batches,
    )


# ------ Registration: upload and live Tkinter ------

@app.route("/register", methods=["GET"])
def page_register():
    """
    Render the student self-registration page.
    Pass the list of semesters as `fetch_semesters` (used by the template).
    """
    semesters = fetch_all("SELECT * FROM semesters ORDER BY semester")
    return render_template("register.html", fetch_semesters=semesters)


@app.route("/api/register_face", methods=["POST"])
def api_register_face():
    """
    Upload-based registration (mobile/camera photo).
    Expects image + display_name + optional usn + semester.
    """
    f = request.files.get("image")
    display_name = request.form.get("display_name","").strip()
    usn = request.form.get("usn","").strip()
    semester = request.form.get("semester","").strip()

    if not f or f.filename == "" or not display_name:
        return jsonify({"error":"image and display_name required"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error":"file type not allowed"}), 400

    filename = secure_filename(f.filename)
    unique = str(uuid.uuid4())
    saved_name = unique + "_" + filename
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    f.save(saved_path)

    # compute embedding using face_utils
    try:
        img = face_utils.load_image_file_bytes(open(saved_path,"rb").read())
        faces = face_utils.detect_faces_rgb(img)
    except Exception as e:
        os.remove(saved_path)
        return jsonify({"error":"invalid image", "detail":str(e)}), 400

    if not faces:
        os.remove(saved_path)
        return jsonify({"error":"no face detected"}), 400

    try:
        embedding = face_utils.compute_embedding_from_bgr(img, faces[0])
    except Exception as e:
        os.remove(saved_path)
        return jsonify({"error":"embedding failed", "detail": str(e)}), 500

    vector_text = face_utils.vector_to_text(embedding)

    # create or update student
    if usn:
        student = fetch_one("SELECT * FROM students WHERE usn=%s", (usn,))
    else:
        student = None
    if student:
        student_id = student["id"]
        execute("UPDATE students SET display_name=%s, semester=%s WHERE id=%s", (display_name, semester, student_id))
    else:
        student_id = execute("INSERT INTO students (display_name, usn, semester) VALUES (%s,%s,%s)",
                             (display_name, usn or None, semester or None))

    execute("DELETE FROM embeddings WHERE student_id=%s", (student_id,))
    execute("INSERT INTO embeddings (student_id, vector_text) VALUES (%s,%s)", (student_id, vector_text))

    # refresh cache
    refresh_embeddings_cache()

    return jsonify({"ok":True, "student_id": student_id})

@app.route("/api/check_usn")
def api_check_usn():
    """
    Returns JSON:
      { exists: true/false, student: {...} }
    Always returns a JSON object with HTTP 200 (unless required param missing).
    This avoids the frontend getting an HTML 404 or plain text response.
    """
    usn = request.args.get("usn", "").strip()
    if not usn:
        return jsonify({"error": "usn required"}), 400

    try:
        student = fetch_one("SELECT id, display_name, usn, semester FROM students WHERE usn=%s", (usn,))
    except Exception as e:
        # DB error -> return JSON error
        return jsonify({"error": "db error", "detail": str(e)}), 500

    if student:
        return jsonify({"exists": True, "student": student}), 200
    else:
        # IMPORTANT: return a JSON object with exists=false, DO NOT return plain text
        return jsonify({"exists": False}), 200


@app.route("/register_live", methods=["POST", "GET"])
def register_live():
    """
    Launches the Tkinter registration GUI script (get_faces_from_camera_tkinter.py) as a subprocess.
    The GUI will run on the server machine (desktop) and allows you to capture multiple faces.
    """
    if request.method == "POST":
        # run the user's Tkinter script in a new process
        script_path = os.path.join(os.getcwd(), "get_faces_from_camera_tkinter.py")
        if not os.path.exists(script_path):
            return jsonify({"error":"registration script not found"}), 500

        # Launch separate Python process (non-blocking)
        try:
            subprocess.Popen([os.getenv("PYTHON_EXECUTABLE","python"), script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return jsonify({"ok":True, "msg":"Registration GUI launched on server"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # GET: page with a button to launch the desktop registration
    return render_template("register_live.html")

@app.route("/process_features", methods=["POST","GET"])
def process_features():
    """
    Run features_extraction_to_csv.py to produce data/features_all.csv from data/data_faces_from_camera/.
    Then refresh embeddings cache automatically.
    """
    script_path = os.path.join(os.getcwd(), "features_extraction_to_csv.py")
    if not os.path.exists(script_path):
        return jsonify({"error":"features extraction script not found"}), 500

    try:
        # blocking call; that's fine for admin use
        subprocess.check_call([os.getenv("PYTHON_EXECUTABLE","python"), script_path])
        refresh_embeddings_cache()
        return jsonify({"ok":True, "msg":"features extracted and embeddings refreshed", "count": len(embeddings_cache)}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error":"feature extraction failed", "detail": str(e)}), 500

# ------ Teacher pages / sessions / attendance ------

@app.route("/teacher", methods=["GET", "POST"])
def page_teacher():
    if request.method == "POST":
        teacher_name = request.form.get("teacher_name", "Teacher").strip()
        t = fetch_one("SELECT * FROM teachers WHERE name=%s", (teacher_name,))
        if t:
            tid = t["id"]
        else:
            tid = execute("INSERT INTO teachers (name) VALUES (%s)", (teacher_name,))

        subject_id = request.form.get("subject_id")
        if not subject_id:
            flash("Select subject")
            return redirect(url_for("page_teacher"))
        session_date = request.form.get("date") or date.today().isoformat()
        batch_id = request.form.get("batch_id") or None
        sid = execute(
            "INSERT INTO sessions (teacher_id, subject_id, date, batch_id) VALUES (%s,%s,%s,%s)",
            (tid, subject_id, session_date, batch_id)
        )
        return redirect(url_for("page_take_attendance", session_id=sid))


    subjects = fetch_all("SELECT * FROM subjects ORDER BY semester,name")
    return render_template("teacher.html", subjects=subjects, today=date.today().isoformat())

@app.route('/take_attendance/<int:session_id>', methods=['GET','POST'])
def page_take_attendance(session_id):
    session = fetch_one("SELECT * FROM sessions WHERE id=%s", (session_id,))
    if not session:
        return "session not found", 404

    subject = fetch_one("SELECT * FROM subjects WHERE id=%s", (session['subject_id'],))
    if not subject:
        return "subject not found", 404

    # If session contains a batch_id, restrict students to that batch's USN range.
    students = None
    batch_id = session.get('batch_id')
    if batch_id:
        batch = fetch_one("SELECT * FROM batches WHERE id=%s", (batch_id,))
        if batch:
            students = fetch_all(
                "SELECT id, display_name, usn FROM students WHERE semester=%s AND usn BETWEEN %s AND %s ORDER BY display_name",
                (subject['semester'], batch['usn_start'], batch['usn_end'])
            )

    if students is None:
        students = fetch_all(
            "SELECT id, display_name, usn FROM students WHERE semester=%s ORDER BY display_name",
            (subject['semester'],)
        )

    if request.method == 'POST':
        # upload-based detection (handled in previous implementations)
        f = request.files.get("image")
        if not f:
            return jsonify({"error":"image required"}), 400
        data = f.read()
        try:
            img = face_utils.load_image_file_bytes(data)
        except Exception as e:
            return jsonify({"error":"invalid image", "detail":str(e)}), 400

        faces = face_utils.detect_faces_rgb(img)
        if not faces:
            student_list = [{"id": s["id"], "name": s["display_name"], "usn": s["usn"]} for s in students]
            return jsonify({"present_ids": [], "attention_scores": {}, "students": student_list, "message":"no faces detected"}), 200

        known_embeddings = [(s["id"], embeddings_cache.get(s["id"])) for s in students if embeddings_cache.get(s["id"])]
        present_ids = set()
        attention_scores = {}
        for face_rect in faces:
            try:
                emb, att_score = face_utils.compute_embedding_and_attention(img, face_rect)
            except Exception:
                continue
            sid, dist = face_utils.match_embedding(known_embeddings, emb, threshold=0.55)
            if sid:
                present_ids.add(sid)
                attention_scores[str(sid)] = att_score

        student_list = [{"id": s["id"], "name": s["display_name"], "usn": s["usn"]} for s in students]
        return jsonify({"present_ids": list(present_ids), "attention_scores": attention_scores, "students": student_list})

    return render_template("take_attendance.html", session=session, subject=subject, students=students)


@app.route("/api/mark_bulk", methods=["POST"])
def api_mark_bulk():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error":"invalid JSON"}), 400

    subject_id = data.get('subject_id')
    dt = data.get('date')
    present_ids = set(map(int, data.get('present_ids', [])))
    batch_id = data.get('batch_id', None)  # optional, may be None
    raw_attention = data.get('attention_scores') or {}
    attention_scores = {}
    if isinstance(raw_attention, dict):
        for k, v in raw_attention.items():
            try:
                if v is not None:
                    attention_scores[int(k)] = round(float(v), 1)
            except (TypeError, ValueError):
                pass

    if not subject_id or not dt:
        return jsonify({"error":"subject_id and date required"}), 400

    subject = fetch_one("SELECT * FROM subjects WHERE id=%s", (subject_id,))
    if not subject:
        return jsonify({"error":"subject not found"}), 404

    # Determine student list: if batch_id provided restrict by USN range; else all semester students
    if batch_id:
        batch = fetch_one("SELECT * FROM batches WHERE id=%s", (batch_id,))
        if not batch:
            return jsonify({"error":"batch not found"}), 404
        students = fetch_all(
            "SELECT id FROM students WHERE semester=%s AND usn BETWEEN %s AND %s",
            (subject['semester'], batch['usn_start'], batch['usn_end'])
        )
    else:
        students = fetch_all("SELECT id FROM students WHERE semester=%s", (subject['semester'],))

    student_ids = [s['id'] for s in students]

    for sid in student_ids:
        if sid in present_ids:
            att_score = attention_scores.get(sid)
            execute(
                "INSERT INTO attendance (date, subject_id, student_id, status, time_marked, attention_score) VALUES (%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE status=VALUES(status), time_marked=VALUES(time_marked), attention_score=VALUES(attention_score)",
                (dt, subject_id, sid, 'Present', datetime.now().strftime('%H:%M:%S'), att_score)
            )
        else:
            execute(
                "INSERT INTO attendance (date, subject_id, student_id, status, attention_score) VALUES (%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE status=VALUES(status), attention_score=VALUES(attention_score)",
                (dt, subject_id, sid, 'Absent', None)
            )

    return jsonify({"ok":True, "present_count": len(present_ids)}), 200


# ------ New: live MJPEG stream & single-frame capture from server webcam ------
import threading as _cam_threading
import time as _cam_time

_camera_lock = _cam_threading.Lock()
_camera_cap = None
_camera_stream_id = 0
_camera_latest_jpeg = None


def _open_server_camera():
    import cv2
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, frm = cap.read()
            if ret and frm is not None:
                return cap, frm
        cap.release()
    return None, None


def _release_server_camera():
    global _camera_cap, _camera_stream_id
    _camera_stream_id += 1
    if _camera_cap is not None:
        try:
            _camera_cap.release()
        except Exception:
            pass
        _camera_cap = None


@app.route("/api/server_camera_stream")
def api_server_camera_stream():
    """Live MJPEG stream from the PC webcam via OpenCV."""
    import cv2
    from flask import Response

    global _camera_cap, _camera_stream_id, _camera_latest_jpeg

    with _camera_lock:
        _release_server_camera()
        cap, first_frame = _open_server_camera()
        if cap is None:
            return jsonify({"error": "Server webcam is unavailable or in use by another app."}), 500
        _camera_cap = cap
        my_id = _camera_stream_id
        ok, buf = cv2.imencode(".jpg", first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            _camera_latest_jpeg = buf.tobytes()

    def generate():
        global _camera_latest_jpeg
        try:
            while True:
                with _camera_lock:
                    if my_id != _camera_stream_id or _camera_cap is None:
                        break
                    ret, frame = _camera_cap.read()
                if not ret or frame is None:
                    _cam_time.sleep(0.03)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    jpg = buf.tobytes()
                    _camera_latest_jpeg = jpg
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                _cam_time.sleep(0.03)
        finally:
            with _camera_lock:
                if my_id == _camera_stream_id:
                    _release_server_camera()

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.route("/api/server_camera_stop", methods=["GET", "POST"])
def api_server_camera_stop():
    """Stop the live PC OpenCV webcam stream and release the device."""
    with _camera_lock:
        _release_server_camera()
    return jsonify({"ok": True})


@app.route("/api/server_camera_snapshot", methods=["GET", "POST"])
def api_server_camera_snapshot():
    """
    Capture a single JPEG frame from the server PC webcam via OpenCV.
    Reuses the latest frame if the live server stream is already active.
    """
    import cv2
    from flask import Response

    with _camera_lock:
        if _camera_cap is not None and _camera_latest_jpeg is not None:
            return Response(_camera_latest_jpeg, mimetype="image/jpeg")

        cap, frame = _open_server_camera()
        if cap is None or frame is None:
            return jsonify({"error": "Server webcam is unavailable or in use by another app."}), 500
        try:
            for _ in range(2):
                ret, frm = cap.read()
                if ret and frm is not None:
                    frame = frm
        finally:
            cap.release()

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return jsonify({"error": "Failed to encode webcam frame."}), 500

    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/capture_frame", methods=["POST"])
def api_capture_frame():
    """
    Capture one frame from server webcam, run detection, return present_ids list.
    Form data expects session_id or subject and date.
    """
    session = None
    session_id = request.form.get("session_id")

    # prefer session_id; else accept subject_id + date
    if session_id:
        session = fetch_one("SELECT * FROM sessions WHERE id=%s", (session_id,))
        if not session:
            return jsonify({"error":"session not found"}), 404
        subject = fetch_one("SELECT * FROM subjects WHERE id=%s", (session['subject_id'],))
        dt = session['date']
    else:
        subject_id = request.form.get("subject_id")
        dt = request.form.get("date") or date.today().isoformat()
        if not subject_id:
            return jsonify({"error":"subject_id or session_id required"}), 400
        subject = fetch_one("SELECT * FROM subjects WHERE id=%s", (subject_id,))
        if not subject:
            return jsonify({"error":"subject not found"}), 404

    # capture single frame from default camera
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error":"camera not available"}), 500
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return jsonify({"error":"failed to capture frame"}), 500

    # process using face_utils expecting BGR image array
    try:
        faces = face_utils.detect_faces_rgb(frame)
    except Exception as e:
        return jsonify({"error":"detection failed", "detail":str(e)}), 500

    # Determine student list (respect session.batch_id if present)
    students = None
    if session and session.get('batch_id'):
        batch = fetch_one("SELECT * FROM batches WHERE id=%s", (session.get('batch_id'),))
        if batch:
            students = fetch_all(
                "SELECT id, display_name, usn FROM students WHERE semester=%s AND usn BETWEEN %s AND %s ORDER BY display_name",
                (subject['semester'], batch['usn_start'], batch['usn_end'])
            )

    if students is None:
        students = fetch_all(
            "SELECT id, display_name, usn FROM students WHERE semester=%s ORDER BY display_name",
            (subject['semester'],)
        )

    if not faces:
        student_list = [{"id": s["id"], "name": s["display_name"], "usn": s["usn"]} for s in students]
        return jsonify({"present_ids": [], "attention_scores": {}, "students": student_list, "message":"no faces detected"}), 200

    known_embeddings = [(s["id"], embeddings_cache.get(s["id"])) for s in students if embeddings_cache.get(s["id"])]
    present_ids = set()
    attention_scores = {}
    for face_rect in faces:
        try:
            emb, att_score = face_utils.compute_embedding_and_attention(frame, face_rect)
        except Exception:
            continue
        sid, dist = face_utils.match_embedding(known_embeddings, emb, threshold=0.55)
        if sid:
            present_ids.add(sid)
            attention_scores[str(sid)] = att_score

    student_list = [{"id": s["id"], "name": s["display_name"], "usn": s["usn"]} for s in students]
    return jsonify({"present_ids": list(present_ids), "attention_scores": attention_scores, "students": student_list})

@app.route("/api/detect_frame", methods=["POST"])
def api_detect_frame():
    """
    Accepts a multipart/form-data 'image' (jpeg/png) from teacher browser,
    runs detection and returns JSON: { present_ids: [...], students: [...], message }
    """
    session = None
    session_id = request.form.get("session_id")
    subject_id = request.form.get("subject_id")
    date_q = request.form.get("date")  # optional

    # identify subject & date (priority session_id)
    if session_id:
        session = fetch_one("SELECT * FROM sessions WHERE id=%s", (session_id,))
        if not session:
            return jsonify({"error": "session not found"}), 404
        subject = fetch_one("SELECT * FROM subjects WHERE id=%s", (session["subject_id"],))
        if not subject:
            return jsonify({"error": "subject not found"}), 404
        dt = session["date"]
    else:
        if not subject_id:
            return jsonify({"error": "subject_id or session_id required"}), 400
        subject = fetch_one("SELECT * FROM subjects WHERE id=%s", (subject_id,))
        if not subject:
            return jsonify({"error": "subject not found"}), 404
        dt = date_q or date.today().isoformat()

    # file image
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "image required"}), 400

    try:
        data = f.read()
        img = face_utils.load_image_file_bytes(data)  # expects BGR numpy array
    except Exception as e:
        return jsonify({"error": "invalid image", "detail": str(e)}), 400

    faces = face_utils.detect_faces_rgb(img)  # returns rectangles

    # determine students list. Prefer session.batch_id (if session_id provided), else accept optional form batch_id
    students = None
    batch_id = None
    if session:
        batch_id = session.get('batch_id')
    else:
        batch_id = request.form.get('batch_id') or None

    if batch_id:
        batch = fetch_one("SELECT * FROM batches WHERE id=%s", (batch_id,))
        if batch:
            students = fetch_all(
                "SELECT id, display_name, usn FROM students WHERE semester=%s AND usn BETWEEN %s AND %s ORDER BY display_name",
                (subject["semester"], batch['usn_start'], batch['usn_end'])
            )

    if students is None:
        students = fetch_all("SELECT id, display_name, usn FROM students WHERE semester=%s ORDER BY display_name",
                             (subject["semester"],))

    student_list = [{"id": s["id"], "display_name": s["display_name"], "usn": s["usn"]} for s in students]

    if not faces:
        return jsonify({"present_ids": [], "attention_scores": {}, "students": student_list, "message": "no faces detected"}), 200

    # Match embeddings & compute attention score per detected face
    known_embeddings = [(s["id"], embeddings_cache.get(s["id"])) for s in students if embeddings_cache.get(s["id"])]
    present_ids = set()
    attention_scores = {}
    for face_rect in faces:
        try:
            emb, att_score = face_utils.compute_embedding_and_attention(img, face_rect)
        except Exception:
            continue
        sid, dist = face_utils.match_embedding(known_embeddings, emb, threshold=0.55)
        if sid:
            present_ids.add(sid)
            attention_scores[str(sid)] = att_score

    return jsonify({"present_ids": list(present_ids), "attention_scores": attention_scores, "students": student_list})


# ------ View attendance (subject+date) ------

@app.route("/view_attendance")
def page_view_attendance():
    subject = request.args.get("subject")
    date_q = request.args.get("date")
    if not (subject and date_q):
        # show simple form to choose subject and date
        return render_template("view_attendance.html", records=None)
    rows = fetch_all(
        """SELECT s.display_name, s.usn, s.semester, sub.name AS subject, a.status, a.time_marked
           FROM attendance a
           JOIN students s ON a.student_id=s.id
           JOIN subjects sub ON a.subject_id=sub.id
           WHERE a.date=%s AND sub.name=%s
           ORDER BY s.usn""",
        (date_q, subject)
    )
    return render_template("view_attendance.html", records=rows, subject=subject, date=date_q)

# ------ Attendance report: filter by semester, subject, date range ------
from datetime import timedelta
# (ensure this import exists near top of app.py)

@app.route("/attendance_report", methods=["GET"])
def page_attendance_report():
    """
    Flexible attendance report:
      - Accepts semester, subject, student filters
      - Accepts quick=today|prev1|prev2 which auto-fills dates
      - view=overview => per-subject summary (present/total) for date(s)
      - view=student  => per-student totals for subject in date range
    """
    semester = request.args.get('semester')
    subject = request.args.get('subject')  # id
    student_id = request.args.get('student_id')
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    quick = request.args.get('quick')  # expected values: today, prev1, prev2
    view = request.args.get('view')  # overview | student | detail(default)

    semesters = fetch_all("SELECT DISTINCT semester FROM subjects ORDER BY semester")

    # apply quick date shortcuts (today, prev1 = yesterday, prev2 = day before yesterday)
    today_iso = date.today().isoformat()
    if quick:
        if quick == 'today':
            date_from = date_to = today_iso
        elif quick == 'prev1':
            d = date.today() - timedelta(days=1)
            date_from = date_to = d.isoformat()
        elif quick == 'prev2':
            d = date.today() - timedelta(days=2)
            date_from = date_to = d.isoformat()

    # Prepare selectors
    subjects = []
    students = []
    records = []
    overall_summary = []  # per-subject counts
    student_summary = []  # per-student summary for a subject (present/total)

    if semester:
        subjects = fetch_all("SELECT * FROM subjects WHERE semester=%s ORDER BY name", (semester,))

    # If view=overview and date_from/date_to present => show per-subject summary across date range
    if semester and view == 'overview' and date_from and date_to:
        # For each subject in semester compute present, total counts, and avg attention score
        overall_summary = fetch_all("""
            SELECT sub.id AS subject_id, sub.name AS subject_name,
                   SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END) AS present_count,
                   SUM(1) AS total_marks,
                   AVG(CASE WHEN a.status='Present' THEN a.attention_score END) AS avg_attention
            FROM attendance a
            JOIN subjects sub ON a.subject_id=sub.id
            WHERE sub.semester=%s AND a.date BETWEEN %s AND %s
            GROUP BY sub.id, sub.name
            ORDER BY sub.name
        """, (semester, date_from, date_to))
        total_present = sum(int(r.get("present_count") or 0) for r in overall_summary)
        total_marks = sum(int(r.get("total_marks") or 0) for r in overall_summary)
        att_vals = [float(r["avg_attention"]) for r in overall_summary if r.get("avg_attention") is not None]
        summary_stats = {
            "present_count": total_present,
            "total_count": total_marks,
            "attendance_pct": round((total_present / total_marks) * 100.0, 1) if total_marks > 0 else None,
            "avg_attention": round(sum(att_vals) / len(att_vals), 1) if att_vals else None,
        }
        return render_template("attendance_report.html",
                               semesters=semesters, subjects=subjects, students=students, records=records,
                               overall_summary=overall_summary, student_summary=student_summary,
                               summary_stats=summary_stats,
                               selected={'semester': semester, 'subject': subject, 'from': date_from, 'to': date_to, 'view': view, 'quick': quick})

    # If view=student and subject specified => per-student totals for that subject (date range required)
    if subject and view == 'student' and date_from and date_to:
        subj = fetch_one("SELECT * FROM subjects WHERE id=%s", (subject,))
        if subj:
            student_summary = fetch_all("""
                SELECT s.id AS student_id,
                       s.display_name,
                       s.usn,
                       COALESCE(SUM(CASE WHEN a.status='Present' THEN 1 ELSE 0 END), 0) AS present_count,
                       COALESCE(COUNT(a.date), 0) AS total_classes,
                       AVG(CASE WHEN a.status='Present' THEN a.attention_score END) AS avg_attention
                FROM students s
                LEFT JOIN attendance a
                  ON a.student_id = s.id
                  AND a.subject_id = %s
                  AND a.date BETWEEN %s AND %s
                WHERE s.semester = %s
                GROUP BY s.id, s.display_name, s.usn
                ORDER BY s.usn
            """, (subject, date_from, date_to, subj['semester']))
            total_present = sum(int(r.get("present_count") or 0) for r in student_summary)
            total_classes = sum(int(r.get("total_classes") or 0) for r in student_summary)
            att_vals = [float(r["avg_attention"]) for r in student_summary if r.get("avg_attention") is not None]
            summary_stats = {
                "present_count": total_present,
                "total_count": total_classes,
                "attendance_pct": round((total_present / total_classes) * 100.0, 1) if total_classes > 0 else None,
                "avg_attention": round(sum(att_vals) / len(att_vals), 1) if att_vals else None,
            }
            return render_template("attendance_report.html",
                                   semesters=semesters, subjects=subjects, students=students, records=records,
                                   overall_summary=overall_summary, student_summary=student_summary,
                                   summary_stats=summary_stats,
                                   selected={'semester': semester, 'subject': subject, 'from': date_from, 'to': date_to, 'view': view, 'quick': quick})

    # Default detail view: works for subject + date range, or semester + date range
    summary_stats = None
    if date_from and date_to and (subject or semester):
        if subject:
            rows = fetch_all(
                """SELECT a.date, s.display_name, s.usn, sub.name AS subject, a.status, a.time_marked, a.attention_score
                   FROM attendance a
                   JOIN students s ON a.student_id=s.id
                   JOIN subjects sub ON a.subject_id=sub.id
                   WHERE a.date BETWEEN %s AND %s AND sub.id=%s
                   ORDER BY a.date DESC, s.usn""",
                (date_from, date_to, subject)
            )
        else:
            rows = fetch_all(
                """SELECT a.date, s.display_name, s.usn, sub.name AS subject, a.status, a.time_marked, a.attention_score
                   FROM attendance a
                   JOIN students s ON a.student_id=s.id
                   JOIN subjects sub ON a.subject_id=sub.id
                   WHERE a.date BETWEEN %s AND %s AND sub.semester=%s
                   ORDER BY a.date DESC, sub.name, s.usn""",
                (date_from, date_to, semester)
            )
        records = rows
        if records:
            total_count = len(records)
            present_count = sum(1 for r in records if r.get("status") == "Present")
            att_vals = [float(r["attention_score"]) for r in records if r.get("status") == "Present" and r.get("attention_score") is not None]
            summary_stats = {
                "present_count": present_count,
                "total_count": total_count,
                "attendance_pct": round((present_count / total_count) * 100.0, 1) if total_count > 0 else None,
                "avg_attention": round(sum(att_vals) / len(att_vals), 1) if att_vals else None,
            }

    return render_template("attendance_report.html",
                           semesters=semesters, subjects=subjects, students=students, records=records,
                           overall_summary=overall_summary, student_summary=student_summary,
                           summary_stats=summary_stats,
                           selected={'semester': semester, 'subject': subject, 'from': date_from, 'to': date_to, 'view': view, 'quick': quick})

@app.route("/admin/batches", methods=["GET","POST"])
def admin_batches():
    # Basic admin page to add/list batches
    if request.method == "POST":
        subject_id = request.form.get("subject_id")
        name = request.form.get("name")
        usn_start = request.form.get("usn_start")
        usn_end = request.form.get("usn_end")
        if subject_id and name and usn_start and usn_end:
            execute("INSERT INTO batches (subject_id, name, usn_start, usn_end) VALUES (%s,%s,%s,%s)",
                    (subject_id, name, usn_start, usn_end))
            flash("Batch added")
        return redirect(url_for("admin_batches"))

    subjects = fetch_all("SELECT * FROM subjects ORDER BY semester, name")
    batches = fetch_all("""SELECT b.id, b.name, b.usn_start, b.usn_end, b.subject_id, s.name AS subject_name, s.semester
                           FROM batches b JOIN subjects s ON s.id=b.subject_id
                           ORDER BY s.semester, s.name, b.name""")
    return render_template("admin_batches.html", subjects=subjects, batches=batches)

@app.route("/api/get_batches/<int:subject_id>")
def api_get_batches(subject_id):
    """Return batches for a given subject."""
    rows = fetch_all(
        "SELECT id, name, usn_start, usn_end FROM batches WHERE subject_id=%s ORDER BY name",
        (subject_id,)
    )
    return jsonify(rows)


# ------ Timetable helper: simple view/edit endpoints ------

@app.route("/timetable", methods=["GET"])
def page_timetable():
    # show timetable entries and subjects
    subjects = fetch_all("SELECT * FROM subjects ORDER BY semester, name")
    timetable = fetch_all("""SELECT t.id, t.day_of_week, t.start_time, t.end_time, s.name, s.semester
                             FROM timetable t JOIN subjects s ON t.subject_id = s.id
                             ORDER BY FIELD(t.day_of_week,'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'), t.start_time""")
    return render_template("timetable.html", timetable=timetable, subjects=subjects)

# --------------------------
# Student management (list / add / edit / delete)
# --------------------------

@app.route("/students", methods=["GET"])
def page_students():
    """
    List students filtered by semester and sortable by USN or name.
    Query params:
      - semester (string) optional
      - sort (usn_asc | usn_desc | name_asc | name_desc) optional
    """
    semester = request.args.get("semester")
    sort = request.args.get("sort", "usn_asc")

    semesters = fetch_all("SELECT DISTINCT semester FROM subjects ORDER BY semester")

    order_clause = "s.usn ASC"
    if sort == "usn_desc":
        order_clause = "s.usn DESC"
    elif sort == "name_asc":
        order_clause = "s.display_name ASC"
    elif sort == "name_desc":
        order_clause = "s.display_name DESC"

    params = ()
    if semester:
        qry = f"SELECT s.id, s.display_name, s.usn, s.semester FROM students s WHERE s.semester=%s ORDER BY {order_clause}"
        students = fetch_all(qry, (semester,))
    else:
        qry = f"SELECT s.id, s.display_name, s.usn, s.semester FROM students s ORDER BY {order_clause}"
        students = fetch_all(qry)

    return render_template("students.html",
                           students=students,
                           semesters=semesters,
                           selected_semester=semester,
                           sort=sort)


@app.route("/students/add", methods=["GET","POST"])
def page_student_add():
    if request.method == "POST":
        display_name = request.form.get("display_name","").strip()
        usn = request.form.get("usn","").strip()
        semester = request.form.get("semester","").strip() or None
        if not display_name or not usn:
            flash("Name and USN are required")
            return redirect(url_for("page_student_add"))
        # avoid duplicate USN
        existing = fetch_one("SELECT id FROM students WHERE usn=%s", (usn,))
        if existing:
            flash("USN already exists, use edit to update.")
            return redirect(url_for("page_students"))
        execute("INSERT INTO students (display_name, usn, semester) VALUES (%s,%s,%s)", (display_name, usn, semester))
        flash("Student added")
        return redirect(url_for("page_students"))
    semesters = fetch_all("SELECT DISTINCT semester FROM subjects ORDER BY semester")
    return render_template("student_add.html", semesters=semesters)


@app.route("/students/edit/<int:student_id>", methods=["GET","POST"])
def page_student_edit(student_id):
    student = fetch_one("SELECT * FROM students WHERE id=%s", (student_id,))
    if not student:
        return "student not found", 404
    if request.method == "POST":
        display_name = request.form.get("display_name","").strip()
        usn = request.form.get("usn","").strip()
        semester = request.form.get("semester","").strip() or None
        if not display_name or not usn:
            flash("Name and USN required")
            return redirect(url_for("page_student_edit", student_id=student_id))
        # check USN duplicate (other student)
        other = fetch_one("SELECT id FROM students WHERE usn=%s AND id<>%s", (usn, student_id))
        if other:
            flash("Another student already has this USN")
            return redirect(url_for("page_student_edit", student_id=student_id))
        execute("UPDATE students SET display_name=%s, usn=%s, semester=%s WHERE id=%s", (display_name, usn, semester, student_id))
        flash("Student updated")
        return redirect(url_for("page_students"))
    semesters = fetch_all("SELECT DISTINCT semester FROM subjects ORDER BY semester")
    return render_template("student_edit.html", student=student, semesters=semesters)


@app.route("/students/delete/<int:student_id>", methods=["POST"])
def api_student_delete(student_id):
    # optionally you may check and delete embeddings and attendance too
    student = fetch_one("SELECT * FROM students WHERE id=%s", (student_id,))
    if not student:
        return jsonify({"error":"student not found"}), 404
    try:
        # delete embeddings row(s)
        execute("DELETE FROM embeddings WHERE student_id=%s", (student_id,))
        # delete attendance records (optional)
        execute("DELETE FROM attendance WHERE student_id=%s", (student_id,))
        # delete student
        execute("DELETE FROM students WHERE id=%s", (student_id,))
        refresh_embeddings_cache()
        return jsonify({"ok":True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# static upload serving (optional)
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# admin endpoint to refresh embeddings
@app.route("/admin/refresh_embeddings")
def route_refresh_embeddings():
    try:
        refresh_embeddings_cache()
        return jsonify({"ok": True, "count": len(embeddings_cache)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# error handlers
@app.errorhandler(413)
def too_large(e):
    return "File too large", 413

@app.errorhandler(404)
def not_found(e):
    return "Not found", 404

if __name__ == "__main__":
    import threading
    import socket
    from werkzeug.serving import make_server

    # Production prototype server on 0.0.0.0:5050
    # We only start this thread if WERKZEUG_RUN_MAIN is "true" (inside the reloader worker)
    # to avoid starting it twice (in the wrapper and the worker).
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        def run_prod():
            # Find all local IPs
            ips = []
            try:
                # Method 1: Connect to internet
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ips.append(s.getsockname()[0])
                s.close()
            except Exception:
                pass
            
            # Method 2: Hostname
            try:
                hostname = socket.gethostname()
                for ip in socket.gethostbyname_ex(hostname)[2]:
                    if ip not in ips and not ip.startswith("127."):
                        ips.append(ip)
            except Exception:
                pass
            
            print("\n" + "="*60)
            print(" PRODUCTION SERVER STARTED")
            print(" If you cannot connect from another device:")
            print(" 1. Make sure both devices are on the same Wi-Fi/Network.")
            print(" 2. Windows Firewall might be blocking 'Python'. allow it.")
            print("    (You may need to allow port 5050 in Advanced Firewall settings)")
            print("-" * 60)
            print(f" Access locally:      http://127.0.0.1:5050")
            for ip in ips:
                print(f" Access from Network: http://{ip}:5050")
            print("="*60 + "\n")

            # Use make_server for threaded execution to avoid app.run conflicts
            try:
                server = make_server('0.0.0.0', 5050, app)
                server.serve_forever()
            except OSError as e:
                print(f"ERROR: Could not start production server on port 5050: {e}")
                print("Try closing other python processes or changing the port.")
        
        t = threading.Thread(target=run_prod)
        t.daemon = True
        t.start()

    # debug True for development only
    app.run(host="127.0.0.1", port=5001, debug=True)
