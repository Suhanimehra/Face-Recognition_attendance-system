import cv2
import os
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load dataset and assign labels
def prepare_data(data_folder_path):
    current_id = 0
    label_ids = {}
    faces = []
    labels = []

    for root, dirs, files in os.walk(data_folder_path):
        for file in files:
            if file.endswith(("png", "jpg", "jpeg")):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces_detected = face_cascade.detectMultiScale(img, 1.1, 5)
                for (x, y, w, h) in faces_detected:
                    roi = img[y:y+h, x:x+w]
                    faces.append(roi)
                    labels.append(id_)

    return faces, labels, label_ids

faces, labels, label_map = prepare_data("dataset")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

with open("labels.pickle", "wb") as f:
    pickle.dump(label_map, f)

print("Training complete ✅")
