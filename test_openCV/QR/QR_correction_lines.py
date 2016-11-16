# -*- coding: utf-8 -*-
# @Time    : 10/11/2016 16:23
# @Author  : Luke
# @Software: PyCharm


import cv2
import numpy as np
import matplotlib.pyplot as plt


img_ori = cv2.imread('test.jpg')
img = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)
# blank = np.zeros(img.shape)
rows,cols = img.shape
img_area = rows*cols


ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

kernal_size = min(img.shape)/40*2+1
erode_kernal = cv2.getStructuringElement(2,(kernal_size,kernal_size))

th1 = cv2.morphologyEx(th1,cv2.MORPH_OPEN,erode_kernal)

lines = cv2.HoughLines(th1,1,10,rows/3)

result = img_ori.copy()
lines = cv2.HoughLines(th1,1,np.pi/180,cols,rows/2)
for line in lines[0]:
    rho = line[0]
    theta = line[1]
    print rho, theta
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
        pt1 = (int(rho / np.cos(theta)), 0)
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
        cv2.line(result, pt1, pt2, (255))
    else:
        pt1 = (0, int(rho / np.sin(theta)))
        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
        cv2.line(result, pt1, pt2, (255), 1)
plt.imshow(result,'gray')
plt.show()

plt.imshow(th1,'gray')
plt.show()


def distances(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def exchange(pt1,pt2):
    return pt2,pt1

'''
以上是通过轮廓找到四个点
'''


pts1 = np.squeeze(contour) #todo 找到四个角点
dist1 = [distances((0,0), pt) for pt in pts1]
ids = np.argsort(dist1)
pts1 = np.array([pts1[i] for i in ids],'float32')
if pts1[1][1] - pts1[0][1] < pts1[2][1] - pts1[0][1]:
    pts1[1],pts1[2] = exchange(pts1[2],pts1[1])

width = distances(pts1[0], pts1[1])
height = distances(pts1[0], pts1[2])
length = max(width,height)
pts2 = np.array([pts1[0],pts1[0]+np.array([0,length]),
        pts1[0]+np.array([length,0]),pts1[0]+np.array([length,length])]
                ,'float32')

M = cv2.getPerspectiveTransform(pts1,pts2)
new_img = cv2.warpPerspective(img_ori,M,(rows+int(pts1[0,0]),cols+int(pts1[0,1])))
new_img = new_img[max(int(pts1[0,1])-10,0):int(pts1[0,1]+length)+10
            ,max(int(pts1[0,0])-10,0):int(pts1[0,0]+length)+10]
plt.subplot(121),plt.imshow(img_ori)
plt.subplot(122),plt.imshow(new_img)
plt.show()