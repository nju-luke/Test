# -*- coding: utf-8 -*-
# @Time    : 10/14/16 18:25
# @Author  : Luke
# @Software: PyCharm
import cv2
import numpy as np
import tflearn
from matplotlib import pyplot as plt

img1 = cv2.imread('11.jpg')
img2 = cv2.imread('22.jpg')
img3 = cv2.imread('33.jpg')

cv2.imshow("fig1.1",img1)
cv2.imshow("fig2.1",img2)
cv2.imshow("fig3.1",img3)

mul = 0

img1_1 = np.asarray(img1,dtype='float')/255
delta = (1-img1_1)*img1_1
if mul != 1:
    mul = (0.5-np.mean(img1_1))/np.mean(delta)
img1_2 = img1_1 + (1-img1_1)*img1_1*mul
cv2.imshow("fig1.2",img1_2)

img1_1 = np.asarray(img2,dtype='float')/255
delta = (1-img1_1)*img1_1
if mul != 1:
    mul = (0.5-np.mean(img1_1))/np.mean(delta)
img1_2 = img1_1 + (1-img1_1)*img1_1*mul
cv2.imshow("fig2.2",img1_2)

img1_1 = np.asarray(img3,dtype='float')/255
delta = (1-img1_1)*img1_1
if mul != 1:
    mul = (0.5-np.mean(img1_1))/np.mean(delta)
img1_2 = img1_1 + (1-img1_1)*img1_1*mul
cv2.imshow("fig3.2",img1_2)

cv2.waitKey()
cv2.destroyAllWindows()

