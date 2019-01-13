import cv2 

#Pre trained model
face_model = cv2.CascadeClassifier('../models/haarcascade_eye_tree_eyeglasses.xml')
video = cv2.VideoCapture(0)
count = 1
while(True):
    status, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print('x: %s| y: %s | w: %s| h: %s'%(x,y,w,h))
        #BGR Color Frame
        color = (0, 255, 0) 
        rectangle_thickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, rectangle_thickness)

    cv2.imshow('captching ...', frame)
    key = cv2.waitKey(3)
    if key == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
