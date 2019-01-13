import os 
import numpy as np 
from PIL import Image
import cv2 
import pickle

face_model = cv2.CascadeClassifier('../models/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
BASE_PATH = '../camera/'
BASE_DIR = os.walk(BASE_PATH)

x_train = []
y_labels = []
person_id = 0
label_and_ids = {}
for root, dirs, files in BASE_DIR:
    if len(dirs) > 0:
        for dir_name in dirs:
            dir_path = BASE_PATH + dir_name
            label = dir_name #Person Name 
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('jpeg'):
                    file_path = dir_path + '/' + file_name
                    gray_image = Image.open(file_path).convert('L')
                    image_np = np.array(gray_image, 'uint8')
                    faces = face_model.detectMultiScale(image_np, scaleFactor= 1.5, minNeighbors=5)
                    if not label in label_and_ids:  
                        label_and_ids[label] = person_id
                        person_id += 1

                    pid = label_and_ids[label]
                    for (x, y, w, h) in faces:
                        face_cords = image_np[y:y+h, x:x+w]
                        x_train.append(face_cords)
                        y_labels.append(pid)

#print(label_and_ids)
#print(x_train)
#print(y_label)
with open('label_and_ids.pickle', 'wb') as f:
    pickle.dump(label_and_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("identity-trainner.yml")
