# -*- coding: utf-8 -*-
# @Time    : 10/31/16 10:13
# @Author  : Luke
# @Software: PyCharm

import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('测试集1交通沃尔玛联名普卡3.jpg')          # queryImage
img2 = cv2.imread('WM10.jpg') # trainImage
img3 = cv2.imread('1交通沃尔玛联名普卡.jpg') # trainImage

# img1 = remove_logo_u(img1)
# img2 = remove_logo_u(img2)
# img3 = remove_logo_u(img3)

# Initiate SIFT detector
sift = cv2.SIFT()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
matches1 = bf.knnMatch(des1,des3, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.69*n.distance:
        good.append(m)
print len(good)
# img12 = drawMatches(img1,kp1,img2,kp2,good)

good = []
for m,n in matches1:
    if m.distance < 0.69*n.distance:
        good.append(m)
print len(good)
#
# img13 = drawMatches(img1,kp1,img3,kp3,good)
#
# plt.figure()
# plt.imshow(img12),plt.show()
# plt.figure()
# plt.imshow(img13),plt.show()