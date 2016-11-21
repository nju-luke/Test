# -*- coding: utf-8 -*-
# @Time    : 10/30/16 14:16
# @Author  : Luke
# @Software: PyCharm
import numpy as np
import cv2

from test_openCV.cards.test_crop_and_filter import remove_logo_u
import matplotlib.pyplot as plt

from test_openCV.cards.test_orb import drawMatches

is_gray = True
img1 = cv2.imread('测试集3交通银行金卡3.jpg',0)          # queryImage
img2 = cv2.imread("3交通银行金卡.jpg",0) # trainImage

rows,cols = img1.shape

img1 = remove_logo_u(img1,is_gray)
img2 = remove_logo_u(img2,is_gray)

# img1 = np.array(core_LBP(img1),'uint8')
# img2 = np.array(core_LBP(img2),'uint8')

# img1 = cv2.Laplacian(img1,-1,ksize=3)
# img2 = cv2.Laplacian(img2,-1,ksize=3)

# img1 = cv2.Canny(img1,100,200)
# img2 = cv2.Canny(img2,100,200)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img1 = clahe.apply(img1)

MIN_MATCH_COUNT = 10


# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SI
# FT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()

def distances(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []

for m,n in matches:
    if m.distance < 0.69*n.distance:
        if distances(kp1[m.queryIdx].pt,kp2[m.trainIdx].pt) < 0.2*max(rows,cols):
            good.append(m)

for mat in good:
    id1 = mat.queryIdx
    id2= mat.trainIdx
    dis = distances(kp1[id1].pt,kp2[id2].pt)

print len(good)
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = drawMatches(img1,kp1,img2,kp2,good,is_gray)

# plt.imshow(img3),plt.show()