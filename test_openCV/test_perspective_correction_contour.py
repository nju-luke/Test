# -*- coding: utf-8 -*-
# @Time    : 16/11/2016 09:47
# @Author  : Luke
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('3.jpg',0)
img = cv2.resize(img,(1000,649))
blank = np.zeros(img.shape,'uint8')+255

cols,rows = img.shape

ret, thresh = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

th = 0.8
contours = [contour for contour in contours if cv2.contourArea(contour) > th*rows*cols]
if len(contours) > 1:
    contours = contours[np.argmin([cv2.contourArea(contour) for contour in contours])]

pts1 = np.zeros((4, 2),'float32')
fxy = [sum(contour) for contour in np.squeeze(contours)]
pts1[0] = np.squeeze(contours)[np.argmin(fxy)]
pts1[3] =  np.squeeze(contours)[np.argmax(fxy)]
fxy = [contour[0] - contour[1] for contour in np.squeeze(contours)]
pts1[2]  = np.squeeze(contours)[np.argmin(fxy)]
pts1[1] =  np.squeeze(contours)[np.argmax(fxy)]

print pts1

width = max(pts1[:, 0]) - pts1[0][0]
height = max(pts1[:, 1]) - pts1[0][1]

pts2 = np.zeros((4, 2),'float32')
pts2[0] = pts1[0]
pts2[1] = pts1[0] + np.array([width, 0])
pts2[2] = pts1[0] + np.array([0, height])
pts2[3] = pts1[0] + np.array([width, height])

print width,height
print pts2

M = cv2.getPerspectiveTransform(pts1,pts2)
new_img = cv2.warpPerspective(img,M,img.shape[::-1])

print min(pts1[:,0]),max(pts1[:,0]),min(pts1[:,1]),max(pts1[:,1])

th = 0.02

new_img = new_img[max(int(min(pts1[:,1])-th*cols),0):min(int(max(pts1[:,1])+th*cols),cols),
                    max(int(min(pts1[:,0])-th*cols),0):min(int(max(pts1[:,0])+th*cols),rows)]

print int(min(pts1[:,1])-th*cols),int(max(pts1[:,1])+th*cols),cols
print int(min(pts1[:,0])-th*rows),int(max(pts1[:,0])+th*rows),rows

##恢复尺寸

plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(new_img,'gray')
plt.show()


cv2.imwrite("ori.jpg",img)
cv2.imwrite("new.jpg",new_img)
