import cv2
import numpy as np
import glob
#img.png is the fgmask
filelist = glob.glob('/home/abhi/acads/sem8/mlt/project/fd/data/images/*')
filelist.sort()
print filelist[1]
done = 0
for files in filelist:
    img=cv2.imread(files)
    print files
    if done > 200:
        break
    done += 1
    f = open('test1.txt')
    for line in f:
        a=line.split();
        if a[0] == files.split('/')[-1]:
            if a[5] == 'person':
                # cv2.rectangle(img,(int(a[1]),int(a[2])),(int(a[3]),int(a[4])),(255,0,0),2)
                gray=img[int(a[2]):int(a[4]) , int(a[1]):int(a[3])]
            # elif a[5] == 'bicycle' or a[5] == 'motorcycle':
            #     cv2.rectangle(img,(int(a[1]),int(a[2])),(int(a[3]),int(a[4])),(0,255,0),2)
            # elif a[5] == 'car':
            #     cv2.rectangle(img,(int(a[1]),int(a[2])),(int(a[3]),int(a[4])),(0,0,255),2)

    # print files
    cv2.imwrite('/home/abhi/acads/sem8/mlt/project/fd/data/cropped/'+files.split("/")[-1].split('.')[0]+'.jpg',gray)
