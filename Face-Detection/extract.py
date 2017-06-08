import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from random import shuffle, seed
from shutil import copyfile

# img = cv2.imread('test1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print img.shape

mypath = "/home/abhi/acads/sem8/mlt/project/fd/Person"
finalpath =  "/home/abhi/acads/sem8/mlt/project/fd/data"


goodfiles = []
imagesLookedUp = 0;
for f in listdir(mypath):
    filename = join(mypath, f)
    if imagesLookedUp > 20000:
        break
    if isfile(filename) and (".jpg" in filename):
        imagesLookedUp += 1
        if imagesLookedUp%1000 == 0:
            print imagesLookedUp
        img = cv2.imread(join(mypath, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (img.shape[0] * img.shape[1] > 10000 ):
            goodfiles.append((filename,join(finalpath, f)))

seed()
shuffle(goodfiles)
print len(goodfiles)
for i in range(0,100):
    # print goodfiles[i][0],goodfiles[i][1]
    copyfile(goodfiles[i][0],goodfiles[i][1])
