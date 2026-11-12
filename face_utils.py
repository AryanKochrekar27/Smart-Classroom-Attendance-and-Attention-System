# face_utils.py
import os
import glob
import bz2
import numpy as np
import cv2
import dlib
import pickle
from PIL import Image

# models path
DLIB_DIR = os.path.join('data', 'data_dlib')
PREDICTOR_PATH = os.path.join(DLIB_DIR, 'shape_predictor_68_face_landmarks.dat')
FACE_RECOG_MODEL_PATH = os.path.join(DLIB_DIR, 'dlib_face_recognition_resnet_model_v1.dat')


def ensure_predictor_model(predictor_path=PREDICTOR_PATH):
    if os.path.exists(predictor_path):
        return predictor_path
    parts = sorted(glob.glob(f"{predictor_path}.bz2.part*"))
    if not parts:
        raise FileNotFoundError(f"Missing {predictor_path} and chunk files {predictor_path}.bz2.part*")
    print(f"[face_utils] Reassembling {predictor_path} from {len(parts)} chunks...")
    decompressor = bz2.BZ2Decompressor()
    tmp_path = f"{predictor_path}.tmp"
    with open(tmp_path, "wb") as out:
        for part_path in parts:
            with open(part_path, "rb") as pf:
                while True:
                    chunk = pf.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(decompressor.decompress(chunk))
    os.replace(tmp_path, predictor_path)
    return predictor_path


ensure_predictor_model(PREDICTOR_PATH)

# dlib initializations
_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(PREDICTOR_PATH)
_face_rec_model = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)

def load_image_file_bytes(file_bytes):
    #  file_bytes: raw bytes (Flask File.read())
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def detect_faces_rgb(img_rgb):
# img_rgb: numpy BGR; convert and detect
    if img_rgb is None:
        return []
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    faces = _detector(gray, 1)
    return faces

def _eye_aspect_ratio(pts):
    # pts: 6 (x, y) coordinates around an eye
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    if c <= 1e-6:
        return 0.0
    return float((a + b) / (2.0 * c))


def compute_attention_from_shape(shape):
    """
    Compute a 0-100 attention score (%) from dlib 68 facial landmarks using:
      - Eye Aspect Ratio (eye openness / alertness)
      - Horizontal yaw symmetry (facing camera vs looking left/right)
      - Vertical pitch alignment (facing forward vs looking down/up)
    """
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=float)

    left_ear = _eye_aspect_ratio(pts[36:42])
    right_ear = _eye_aspect_ratio(pts[42:48])
    avg_ear = (left_ear + right_ear) / 2.0
    eye_score = float(np.clip((avg_ear - 0.14) / (0.27 - 0.14), 0.0, 1.0))

    nose_x, nose_y = pts[30]
    left_face_x = pts[0][0]
    right_face_x = pts[16][0]
    left_w = abs(nose_x - left_face_x)
    right_w = abs(right_face_x - nose_x)
    denom = max(left_w + right_w, 1.0)
    yaw_asym = abs(left_w - right_w) / denom
    yaw_score = float(np.clip(1.0 - (yaw_asym / 0.45), 0.0, 1.0))

    eye_y = (pts[36][1] + pts[45][1]) / 2.0
    chin_y = pts[8][1]
    face_h = max(chin_y - eye_y, 1.0)
    nose_ratio = (nose_y - eye_y) / face_h
    pitch_dev = max(0.0, abs(nose_ratio - 0.40) - 0.10)
    pitch_score = float(np.clip(1.0 - (pitch_dev / 0.25), 0.0, 1.0))

    score = (0.45 * eye_score + 0.35 * yaw_score + 0.20 * pitch_score) * 100.0
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def compute_embedding_from_bgr(img_bgr, face_rect):
# face_rect: dlib.rectangle
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape = _predictor(rgb, face_rect)
    face_descriptor = _face_rec_model.compute_face_descriptor(rgb, shape)
    return np.array(face_descriptor, dtype=float)


def compute_embedding_and_attention(img_bgr, face_rect):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape = _predictor(rgb, face_rect)
    face_descriptor = _face_rec_model.compute_face_descriptor(rgb, shape)
    attention = compute_attention_from_shape(shape)
    return np.array(face_descriptor, dtype=float), attention

def vector_to_text(vec):
    return ','.join(map(str, vec.tolist()))

def text_to_vector(text):
    if not text:
        return np.zeros(128, dtype=float)
    parts = list(map(float, text.split(',')))
    return np.array(parts, dtype=float)

def match_embedding(known_embeddings, probe_vec, threshold=0.55):
# known_embeddings: list of (student_id, vector_text)
    best = None
    best_dist = 999
    probe = probe_vec
    for sid, vec_text in known_embeddings:
        candidate = text_to_vector(vec_text)
        dist = np.linalg.norm(candidate - probe)
        if dist < best_dist:
            best_dist = dist
            best = sid
    if best is not None and best_dist <= threshold:
        return best, best_dist
    return None, best_dist




if __name__ == '__main__':
    print('face_utils ready')