
import skimage.data
import sys
import os
from skimage.feature import hog
import numpy
import scipy.cluster.vq as vq
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import itertools
import cPickle as pickle
from cPickle import dump, HIGHEST_PROTOCOL
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('/home/gpuuser/Downloads/selectivesearch-develop')
import glob
import selectivesearch
inputPath="/home/gpuuser/frame/"
vidName='nov92015-1'
import cv2
def main():
    img=cv2.imread('frame366.jpg')
    maxInd=max(img.shape[1],img.shape[0])
    scale=maxInd/500.0
    small = cv2.resize(img, (int((img.shape[1])/scale),int((img.shape[0]/scale))))
    cv2.imwrite('imagesmall.jpg',small)
    img_lbl, regions = selectivesearch.selective_search(small, scale=500, sigma=0.9, min_size=30)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = set()
    NoneData=[]
    count=0;
    for r in regions:
        if r['rect'] in candidates:
            continue 
        x, y, w, h = r['rect']
        if w<30 or h<30:
            continue
        candidates.add(r['rect'])
        cv2.rectangle(small,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imwrite('1.jpg',small)

if __name__ == "__main__":
    main()
		

