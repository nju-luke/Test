# -*- coding: utf-8 -*-
# @Time    : 11/2/16 17:46
# @Author  : Luke
# @Software: PyCharm

import numpy as np
import cv2
from test_openCV.cards.test_crop_and_filter import remove_logo_u
from test_openCV.test_orb import matched_dis, drawMatches
import matplotlib.pyplot as plt


is_gray = True
img1 = cv2.imread('测试集1交通沃尔玛联名普卡3.jpg',0)          # queryImage
img2 = cv2.imread("1交通沃尔玛联名普卡_1.jpg",0) # trainImage


# img1 = remove_logo_u(img1,is_gray)
# img2 = remove_logo_u(img2,is_gray)

MIN_MATCH_COUNT = 10

Laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

img11=cv2.Laplacian(img1,cv2.CV_64F,ksize=3)
img21=cv2.Laplacian(img2,cv2.CV_64F,ksize=3)

plt.subplot(221),plt.imshow(img1,'gray')
plt.subplot(222),plt.imshow(img11,'gray')
plt.subplot(223),plt.imshow(img2,'gray')
plt.subplot(224),plt.imshow(img21,'gray')
plt.show()

print img11
