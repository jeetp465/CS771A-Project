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
from sklearn.metrics import jaccard_similarity_score
import os
K_THRESH=1
tot_des=0
nbins = 9
all_features_hog = []
all_features_sift = []
all_features_label_hog = []
all_features_label_sift = []
threshold = 10000
timer=0
import random
from rect import Rect
def computeHistograms(codebook, descriptors):
	code, dist = vq.vq(descriptors, codebook)
	histogram_of_words, bin_edges = histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
	return histogram_of_words

with open('/home/gpuuser/frame/SIFT/codebooks/code1280000_1000.file','rb') as f:
	codebk=pickle.load(f)
def getScore(x1,y1,x2,y2, x11,y11,x12,y12):
    box={};
    box["x"],box["y"]=x1,y1
    box["width"],box["height"]=x2-x1,y2-y1
    rect1 = Rect(box)
    box["x"],box["y"]=x11,y11
    box["width"],box["height"]=x12-x11,y12-y11
    rect2 = Rect(box)
    return rect1.overlap(rect2)

eps=0.0001
thresh=1000
dictcount={}
def main():
    f=open(inputPath+"frame_data/"+vidName+".txt")
    dict1={};
    threhold=0.25
    for line in f:
        frame=line.split()[0]
        data=(line.split()[1:])
        if int(frame) in dict1:
            dict1[int(frame)]+=data
        else:
            dict1[int(frame)]=data
    dictcount[0]=0;
    dictcount[1]=0;
    dictcount[2]=0;
    countn1=0
    for frame,data in dict1.iteritems():
        if os.path.exists(inputPath+vidName+'/frame'+str(frame)+'.jpg'):
            print frame
            img=cv2.imread(inputPath+vidName+'/frame'+str(frame)+'.jpg')
            maxInd=max(img.shape[1],img.shape[0])
            scale=maxInd/500.0
            small = cv2.resize(img, (int((img.shape[1])/scale),int((img.shape[0]/scale))))
            img_lbl, regions = selectivesearch.selective_search(small, scale=500, sigma=0.9, min_size=20)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            candidates = set()
            #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            #ax.imshow(small)
            NoneData=[]
            count=0;
            for r in regions:
                # excluding same rectangle (with different segments)
                if r['rect'] in candidates:
                    continue 
                x, y, w, h = r['rect']
                if w<10 or h<10:
                    continue
                candidates.add(r['rect'])
                maxlabel,maxInd=0.0,0.0
                for i in range(len(data)/5):
                    labels=data[5*i:5*i+5]
                    l1=getScore(int(x*scale),int(y*scale),int(x*scale+w*scale),int(y*scale+h*scale),int(labels[0]),int(labels[1]),int(labels[2]),int(labels[3]));
                    if l1 > maxInd:
                        maxlabel,maxInd=labels[4],l1
                if maxInd < eps:
                    NoneData.append([int(x*scale),int(y*scale),int(x*scale+w*scale),int(y*scale+h*scale)])
            for i in range(len(data)/5):
                label=data[5*i:5*i+5]
                if label[-1] == 'Person' and dictcount[0] < threshold:
                    gray=img[int(label[1]):int(label[3]) , int(label[0]):int(label[2])]
                    gray = cv2.resize(gray,(100, 100), interpolation = cv2.INTER_CUBIC)
                    vector = hog(gray, orientations=250, pixels_per_cell=(50, 50), cells_per_block=(1, 1), visualise=False)
                    all_features_hog.append(vector)
                    all_features_label_hog.append(0)
                    sift = cv2.SIFT()
                    kp, des = sift.detectAndCompute(gray,None)
                    if des is not None:
                        vector=computeHistograms(codebk, des)
                        all_features_sift.append(vector)
                        all_features_label_sift.append(0)
                    count+=1;
                    dictcount[0]+=1;

                elif (label[-1] == 'Bicycle' or label[-1] == 'Motorcycle') and dictcount[1] < threshold :
                    gray=img[int(label[1]):int(label[3]) , int(label[0]):int(label[2])]
                    gray = cv2.resize(gray,(100, 100), interpolation = cv2.INTER_CUBIC)
                    vector = hog(gray, orientations=250, pixels_per_cell=(50, 50), cells_per_block=(1, 1), visualise=False)
                    all_features_hog.append(vector)
                    all_features_label_hog.append(1)
                    sift = cv2.SIFT()
                    kp, des = sift.detectAndCompute(gray,None)
                    if des is not None:
                        vector=computeHistograms(codebk, des)
                        all_features_sift.append(vector)
                        all_features_label_sift.append(1)
                    count+=1;
                    dictcount[1]+=1;
                elif label[-1] == 'Car' and dictcount[2] < threshold:
                    gray=img[int(label[1]):int(label[3]) , int(label[0]):int(label[2])]
                    gray = cv2.resize(gray,(100, 100), interpolation = cv2.INTER_CUBIC)
                    vector = hog(gray, orientations=250, pixels_per_cell=(50, 50), cells_per_block=(1, 1), visualise=False)
                    all_features_hog.append(vector)
                    all_features_label_hog.append(2)
                    sift = cv2.SIFT()
                    kp, des = sift.detectAndCompute(gray,None)
                    if des is not None:
                        vector=computeHistograms(codebk, des)
                        all_features_sift.append(vector)
                        all_features_label_sift.append(2)
                    count+=1;
                    dictcount[2]+=1;
            array1= range(len(NoneData))
            random.shuffle(array1)
            for i in range((2*count)):
                gray=img[int(NoneData[i][1]):int(NoneData[i][3]) , int(NoneData[i][0]):int(NoneData[i][2])]
                gray = cv2.resize(gray,(100, 100), interpolation = cv2.INTER_CUBIC)
                vector = hog(gray, orientations=250, pixels_per_cell=(50, 50), cells_per_block=(1, 1), visualise=False)
                all_features_hog.append(vector)
                all_features_label_hog.append(3)
                sift = cv2.SIFT()
                kp, des = sift.detectAndCompute(gray,None)
                if des is not None:
                    vector=computeHistograms(codebk, des)
                    all_features_sift.append(vector)
                    all_features_label_sift.append(3)
            countn1+=1;
            if countn1 > thresh:
                break
    print(len(all_features_hog))
    print(len(all_features_label_hog))
    print(len(all_features_sift))
    print(len(all_features_label_sift))

    print('saving hog histograms') 
    with open('hog_all1','wb') as f:
	pickle.dump(all_features_hog,f)
    with open('Hog_labels_all1','wb') as f1:
	pickle.dump(all_features_label_hog,f1)
    print('saving sift histograms') 
    with open('sift_all1','wb') as f:
	pickle.dump(all_features_sift,f)
    with open('Sift_labels_all1','wb') as f1:
	pickle.dump(all_features_label_sift,f1)
    print('saving histograms done') 


if __name__ == "__main__":
    main()
		

