# -*- coding: utf-8 -*-
# @Time    : 10/30/16 16:13
# @Author  : Luke
# @Software: PyCharm
import cv2

from test_openCV.cards.test_crop_and_filter import remove_logo_u
import matplotlib.pyplot as plt
import numpy as np
from test_openCV.test_orb import drawMatches

is_gray = True
img1 = cv2.imread('测试集3广发金卡3.jpg',0)          # queryImage
img2 = cv2.imread("0000024金穗携程金卡.jpg",0) # trainImage

rows,cols = img1.shape

img1 = remove_logo_u(img1,is_gray)
img2 = remove_logo_u(img2,is_gray)

surf = cv2.SURF(500)
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

def distances(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)


threshold = 0.69
fine_matches = []
for m, n in matches:
    if m.distance < threshold * n.distance:
        if distances(kp1[m.queryIdx].pt,kp2[m.trainIdx].pt) < 0.2*max(rows,cols):
            fine_matches.append(m)

print len(fine_matches)
img3 = drawMatches(img1,kp1,img2,kp2,fine_matches,is_gray)
