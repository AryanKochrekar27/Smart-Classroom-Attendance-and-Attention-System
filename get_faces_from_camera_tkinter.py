# (REPLACE the create_face_folder, GUI_info, and GUI_get_input_name parts or whole file with the code below)
# Only the new/changed file content shown here:

import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0

        self.win = tk.Tk()
        self.win.title("Face Register")
        self.win.geometry("1000x520")

        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # Right side info
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_usn = tk.Entry(self.frame_right_info)
        self.input_sem = tk.Entry(self.frame_right_info)
        self.input_name_char = ""
        self.input_usn_char = ""
        self.input_sem_char = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        # frame/ROI variables
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)

    def GUI_clear_data(self):
        # Delete face folders and features_all.csv
        folders_rd = []
        if os.path.isdir(self.path_photos_from_camera):
            folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        # Read Name / USN / Semester and create folder
        self.input_name_char = self.input_name.get().strip()
        self.input_usn_char = self.input_usn.get().strip()
        self.input_sem_char = self.input_sem.get().strip()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="Face register",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=10)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: Clear face photos").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=8)
        tk.Button(self.frame_right_info,
                  text='Clear',
                  command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input name, USN, Semester and create folders for face
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: Input student details").grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=8)

        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="USN: ").grid(row=9, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_usn.grid(row=9, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Semester: ").grid(row=10, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_sem.grid(row=10, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                  text='Create Folder (Input)',
                  command=self.GUI_get_input_name).grid(row=11, column=0, columnspan=3, padx=5, pady=6)

        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 3: Save face image").grid(row=12, column=0, columnspan=3, sticky=tk.W, padx=5, pady=8)

        tk.Button(self.frame_right_info,
                  text='Save current face',
                  command=self.save_current_face).grid(row=13, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.log_all.grid(row=14, column=0, columnspan=20, sticky=tk.W, padx=5, pady=8)

        self.frame_right_info.pack()

    def pre_work_mkdir(self):
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.isdir(self.path_photos_from_camera):
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        if os.listdir(self.path_photos_from_camera):
            person_list = os.listdir(self.path_photos_from_camera)
            person_num_list = []
            for person in person_list:
                try:
                    # folder name format: person_{n}_...
                    person_order = person.split('_')[1]
                    person_num_list.append(int(person_order))
                except Exception:
                    continue
            if person_num_list:
                self.existing_faces_cnt = max(person_num_list)
            else:
                self.existing_faces_cnt = 0
        else:
            self.existing_faces_cnt = 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        self.existing_faces_cnt += 1
        # Build folder name: person_{n}_{Name}_{USN}_{SemX}
        name_part = self.input_name_char if self.input_name_char else "unknown"
        usn_part = ("USN" + self.input_usn_char) if self.input_usn_char else ""
        sem_part = ("Sem" + self.input_sem_char) if self.input_sem_char else ""
        # join only non-empty parts
        extra = "_".join([p for p in [name_part, usn_part, sem_part] if p])
        if extra:
            self.current_face_dir = os.path.join(self.path_photos_from_camera,
                                                 "person_" + str(self.existing_faces_cnt) + "_" + extra)
        else:
            self.current_face_dir = os.path.join(self.path_photos_from_camera,
                                                 "person_" + str(self.existing_faces_cnt))
        os.makedirs(self.current_face_dir, exist_ok=True)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)
        self.ss_cnt = 0
        self.face_folder_created_flag = True

    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                   np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                self.face_ROI_width_start - self.ww + jj]
                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.ss_cnt) + ".jpg\"" + " saved!"
                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(self.current_face_dir, "img_face_" + str(self.ss_cnt) + ".jpg"),
                                self.face_ROI_image)
                    logging.info("%-40s %s/img_face_%s.jpg", "Save into：",
                                 str(self.current_face_dir), str(self.ss_cnt) + ".jpg")
                else:
                    self.log_all["text"] = "Please do not out of range!"
            else:
                self.log_all["text"] = "No face in current frame!"
        else:
            self.log_all["text"] = "Please run step 2!"

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640, 480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            if len(faces) != 0:
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()

if __name__ == '__main__':
    main()
