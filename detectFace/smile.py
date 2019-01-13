import cv2 

video = cv2.VideoCapture(0)
smile_detect = cv2.CascadeClassifier('../models/haarcascade_smile.xml')
#smile_detect = cv2.CascadeClassifier('../models/haarcascade_frontalface_alt2.xml')
while(True):
    status, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    smiles = smile_detect.detectMultiScale(gray)
    for (x, y, w, h) in smiles:
        thickness = 2
        color = (255, 0, 0) #BGR
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, thickness)

    cv2.imshow('smiles..', frame)
    cv2.waitKey(2)

video.release()
cv2.destroyAllWindows()
