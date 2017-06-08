import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
person =  "/home/abhi/acads/sem8/mlt/project/fd/data/person"
nonperson = "/home/abhi/acads/sem8/mlt/project/fd/data/nonperson"
finalpath =  "/home/abhi/acads/sem8/mlt/project/fd/data"
hard = "/home/abhi/acads/sem8/mlt/project/fd/data/hard"
finalPerson = "/home/abhi/acads/sem8/mlt/project/fd/data/cropped/person"
finalNonPerson = "/home/abhi/acads/sem8/mlt/project/fd/data/cropped/nonperson"
def check(mydir, imgtype, scaleFactor, minNeighbors):
    correct, total = 0, 0
    for f in listdir(mydir):
        filename = join(mydir, f)
        if isfile(filename) and (".jpg" in filename):
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
            total += 1
            if total%2 == 1:
                print total
            # if len(faces):
            #     for (x,y,w,h) in faces:
            #         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #     cv2.imshow('img',img)
            #     cv2.waitKey(0)
            #     cv2.imwrite('my.png',img)
            #     cv2.destroyAllWindows()
            if imgtype:
                correct += (1 if len(faces) >= 1 else 0)
            else:
                correct += (1 if len(faces) == 0 else 0)
    return (correct, total)

# img = cv2.imread('test.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# print len(faces)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #roi_gray = gray[y:y+h, x:x+w]
    #roi_color = img[y:y+h, x:x+w]
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == '__main__':
    # print check(person, 1, 1.1, 3)
    # print check(nonperson, 0, 1.1, 3)
    # print check(hard, 1, 1.1, 3)
    #1 for person

    print check(finalPerson, 1, 1.1, 3)
    print check(finalNonPerson, 0, 1.1, 3)
