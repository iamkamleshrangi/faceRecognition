import cv2 

#This portion will read the image with opencv2 
path = 'penguins.jpg'
#If second parameter is 0 then it's a gray scrale image otherwise.
#It's a 1,2,3 shows it's colored image file.
img = cv2.imread(path, 2)
#This with give you numpy array of the images file
print(img)
#To show an image
img = cv2.imshow('label', img)
#wait till any key presses 
cv2.waitKey(0)
#It will close all the open windos of cv2
cv2.destroyAllWindows()
