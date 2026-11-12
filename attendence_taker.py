# Replace attendance_taker.py with this updated version:

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

# Dlib initializations
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create / open a connection to the database and ensure table exists
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
table_name = "attendance"
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    name TEXT,
    usn TEXT,
    semester TEXT,
    subject TEXT,
    time TEXT,
    date DATE,
    UNIQUE(name, date, subject)
)
"""
cursor.execute(create_table_sql)
conn.commit()
conn.close()

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.frame_cnt = 0
        self.face_features_known_list = []
        # now store list of dicts: [{'display_name':..,'usn':..,'semester':..}, ...]
        self.face_name_known_list = []

        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []

        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # subject for this session (prompt user)
        self.subject = ""
        self.get_subject_for_session()

    def get_subject_for_session(self):
        # Simple prompt for subject - you can replace with a GUI or arg as desired
        subj = input("Enter subject name for this attendance session (e.g., DBMS): ").strip()
        if not subj:
            subj = "UNKNOWN_SUBJECT"
        self.subject = subj

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=0)
            for i in range(csv_rd.shape[0]):
                # first three columns: display_name, usn, semester
                display_name = str(csv_rd.iloc[i, 0])
                usn = str(csv_rd.iloc[i, 1]) if not pd.isna(csv_rd.iloc[i, 1]) else ""
                semester = str(csv_rd.iloc[i, 2]) if not pd.isna(csv_rd.iloc[i, 2]) else ""
                self.face_name_known_list.append({
                    "display_name": display_name,
                    "usn": usn,
                    "semester": semester
                })
                # features are in columns 3..130
                features_someone_arr = []
                for j in range(3, 131):
                    val = csv_rd.iloc[i, j]
                    # handle missing
                    if pd.isna(val) or val == '':
                        features_someone_arr.append(0.0)
                    else:
                        features_someone_arr.append(float(val))
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run registration and feature extraction before starting recognition")
            return 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1, dtype=float)
        feature_2 = np.array(feature_2, dtype=float)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                e_distance_current_frame_person_x_list.append(self.last_current_frame_centroid_e_distance)
            if e_distance_current_frame_person_x_list:
                last_frame_num = e_distance_current_frame_person_x_list.index(min(e_distance_current_frame_person_x_list))
                self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"Subject: {self.subject}", (20, 190), self.font, 0.8, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            pos = (int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1]))
            cv2.putText(img_rd, str(self.current_frame_face_name_list[i]), pos, self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def attendance(self, person_info):
        # person_info: dict with keys display_name, usn, semester
        display_name = person_info.get("display_name", "")
        usn = person_info.get("usn", "")
        semester = person_info.get("semester", "")
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ? AND subject = ?", (display_name, current_date, self.subject))
        existing_entry = cursor.fetchone()
        if existing_entry:
            print(f"{display_name} already marked present for {current_date} under {self.subject}")
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT OR IGNORE INTO attendance (name, usn, semester, subject, time, date) VALUES (?, ?, ?, ?, ?, ?)",
                           (display_name, usn, semester, self.subject, current_time, current_date))
            conn.commit()
            print(f"{display_name} marked present for {current_date} | {self.subject} at {current_time}")
        conn.close()

    def process(self, stream):
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                if not flag:
                    break
                kk = cv2.waitKey(1)
                faces = detector(img_rd, 0)

                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1: No face cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1
                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append([int((faces[k].left() + faces[k].right())/2), int((faces[k].top() + faces[k].bottom())/2)])
                            img_rd = cv2.rectangle(img_rd, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 2)
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()
                    for i in range(self.current_frame_face_cnt):
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                    self.draw_note(img_rd)
                else:
                    logging.debug("scene 2: Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    if self.current_frame_face_cnt == 0:
                        logging.debug("No faces in this frame")
                        self.current_frame_face_name_list = []
                    else:
                        logging.debug("Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        for k in range(len(faces)):
                            logging.debug("For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append([int((faces[k].left() + faces[k].right())/2), int((faces[k].top() + faces[k].bottom())/2)])
                            self.current_frame_face_position_list.append((faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)))
                            self.current_frame_face_X_e_distance_list = []

                            for i in range(len(self.face_features_known_list)):
                                e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k], self.face_features_known_list[i])
                                self.current_frame_face_X_e_distance_list.append(e_distance_tmp)

                            similar_person_num = self.current_frame_face_X_e_distance_list.index(min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                person_info = self.face_name_known_list[similar_person_num]
                                disp_name = person_info.get("display_name", "unknown")
                                self.current_frame_face_name_list[k] = disp_name
                                logging.debug("Face recognition result: %s", disp_name)
                                # mark attendance using the person_info dict
                                self.attendance(person_info)
                            else:
                                logging.debug("Face recognition result: Unknown person")

                        self.draw_note(img_rd)

                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)
                logging.debug("Frame ends\n\n")

    def run(self):
        cap = cv2.VideoCapture(0)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()

if __name__ == '__main__':
    main()
