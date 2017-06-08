import numpy as np
import cv2

i = 1
i=301
while(1):
  fgbg = cv2.BackgroundSubtractorMOG()
  while i < 303:
    print 'images/frame' + str(i) + '.jpg'
    frame = cv2.imread('images/frame' + str(i) + '.jpg')
    if frame == None:
        i += 1
        continue
    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

  cv2.destroyAllWindows()
