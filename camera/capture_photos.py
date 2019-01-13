import cv2, uuid
import os, re, time

def main(camera=0):
    #Number photos
     number_photos = 10
    #Interval between 1st and 2nd photo capture
    time_interval = 1
    #Photo counter
    count = 1
    video = cv2.VideoCapture(camera)
    print('Enter Your Name:')
    name = input()
    dir_name = dirName(name)
    while True:
        time.sleep(time_interval)
        status, frame = video.read() 
        saveImages(dir_name, frame)
        print('Photo Count = %s'%(count))
        count += 1
        key = cv2.waitKey(1)
        if count == number_photos:
            break
    cv2.destroyAllWindows()
            
def dirName(name):
    dir_name = re.sub('\s+','_', name)
    return dir_name

def saveImages(dir_name, frame):
    try:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        file_name = str(uuid.uuid4().hex) + '.jpg'
        path = dir_name + "/" + file_name
        cv2.imwrite(path, frame)
        return True, path 
    except Exception as e:
        print('Exception => %s'%e)
        return False, ''
main()
        
