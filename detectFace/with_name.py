import cv2 
import numpy as np 
import pickle

#Pre trained model
face_model = cv2.CascadeClassifier('../models/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../customized_models/identity-trainner.yml')

labels = {}
with open('../customized_models/label_and_ids.pickle', 'rb') as f: 
    lds  = pickle.load(f)
    labels = {v:k for k,v in lds.items()}

print(labels)
video = cv2.VideoCapture(0)
count = 1
while(True):
    status, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print('x: %s| y: %s | w: %s| h: %s'%(x,y,w,h))
        gray_cords = gray[y:y+h, x:x+w]
        color_cords = frame[y:y+h, x:x+w]
        #BGR Color Frame
        pid, conf = recognizer.predict(gray_cords)
        if conf >= 35:
            person_name = labels[pid]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0,0,255) #BGR
            thickness = 2 
            cv2.putText(frame, person_name, (x, y), font, 1, color, thickness, cv2.LINE_AA)

        color = (0, 255, 0) 
        rectangle_thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, rectangle_thickness)

    cv2.imshow('captching ...', frame)
    key = cv2.waitKey(3)
    if key == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
