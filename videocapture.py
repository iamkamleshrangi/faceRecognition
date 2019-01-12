import cv2 
import time 

video = cv2.VideoCapture(0)
status, frame = video.read()
print(frame)
print(status)
time.sleep(3)
video.release()
