# Replace your current features_extraction_to_csv.py with this updated version:

import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Dlib initializations
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)
    logging.info("%-40s %-20s", " Image with faces detected:", path_img)
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face in %s", path_img)
    return face_descriptor

def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = [f for f in os.listdir(path_face_personX) if os.path.isfile(os.path.join(path_face_personX, f))]
    if photos_list:
        for img_name in photos_list:
            logging.info("%-40s %-20s", " / Reading image:", path_face_personX + "/" + img_name)
            features_128d = return_128d_features(os.path.join(path_face_personX, img_name))
            if features_128d == 0:
                continue
            else:
                features_list_personX.append(features_128d)
    else:
        logging.warning(" Warning: No images in %s/", path_face_personX)

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=float).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=float)
    return features_mean_personX

def parse_folder_name(folder_name):
    # expected formats:
    # person_1_Name_USN123_Sem5
    # person_1_Name
    parts = folder_name.split('_')
    display_name = ""
    usn = ""
    semester = ""
    # remove leading 'person' and index
    if len(parts) >= 3:
        # everything after person_{n}_ is candidate info
        remaining = parts[2:]
        # heuristics: look for tokens starting with USN or Sem
        for tok in remaining:
            if tok.upper().startswith("USN"):
                usn = tok
            elif tok.upper().startswith("SEM"):
                semester = tok
            else:
                # append to display name
                if display_name:
                    display_name += "_" + tok
                else:
                    display_name = tok
    elif len(parts) == 2:
        display_name = parts[1]
    else:
        display_name = folder_name
    return display_name, usn, semester

def main():
    logging.basicConfig(level=logging.INFO)
    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    out_csv = "data/features_all.csv"
    # write header
    header = ["display_name", "usn", "semester"] + [f"f{i}" for i in range(1,129)]
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for person in person_list:
            person_dir = os.path.join(path_images_from_camera, person)
            if not os.path.isdir(person_dir):
                continue
            logging.info("Processing folder: %s", person_dir)
            features_mean_personX = return_features_mean_personX(person_dir)
            display_name, usn, semester = parse_folder_name(person)
            row = [display_name, usn, semester] + list(np.round(features_mean_personX.astype(float), 6))
            writer.writerow(row)
            logging.info("Wrote features for %s", person)
    logging.info("Save all the features of faces registered into: %s", out_csv)

if __name__ == '__main__':
    main()
