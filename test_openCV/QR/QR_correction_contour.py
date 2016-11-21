# -*- coding: utf-8 -*-
# @Time    : 10/11/2016 16:23
# @Author  : Luke
# @Software: PyCharm


import cv2
import numpy as np
import matplotlib.pyplot as plt


img_ori = cv2.imread('test3.jpg')                #读取图片
img = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)  # 转换成灰度图
img = cv2.medianBlur(img,5)                     #模糊处理
# blank = np.zeros(img.shape)
rows,cols = img.shape
img_area = rows*cols


ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)  #二值化

kernal_size = min(img.shape)/40*2+1
erode_kernal = cv2.getStructuringElement(2,(kernal_size,kernal_size))   #opencv开运算运算核

th1 = cv2.morphologyEx(th1,cv2.MORPH_OPEN,erode_kernal)                 #opencv开运算
contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   #寻找轮廓

area_QR = np.Inf                                    #通过面积比较，删除小于阈值的轮廓，找到符合标准的
for i in range(len(contours))[::-1]:
    area_i = cv2.contourArea(contours[i])
    if area_i > img_area/3 and area_i < area_QR:
        contour = contours[i]
        area_QR = area_i
epsilon = 0.1 * cv2.arcLength(contour, True)
contour = cv2.approxPolyDP(contour, epsilon, True)
#
# for i in range(len(contours))[::-1]:
#     print cv2.contourArea(contours[i]),img_area
#     if cv2.contourArea(contours[i]) < img_area/3:
#         contours.pop(i)

# cv2.drawContours(blank, contour, -1, 255)
# blank = np.array(blank,'uint8')

def distances(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def exchange(pt1,pt2):
    return pt2,pt1

'''
以上是通过轮廓找到四个点
'''

pts1 = np.squeeze(contour)                                  #定位四个点
dist1 = [distances((0,0), pt) for pt in pts1]
ids = np.argsort(dist1)
pts1 = np.array([pts1[i] for i in ids],'float32')
if pts1[1][1] - pts1[0][1] < pts1[2][1] - pts1[0][1]:
    pts1[1],pts1[2] = exchange(pts1[2],pts1[1])

width = distances(pts1[0], pts1[1])                         #确定透视关系以后的新的4个点的位置
height = distances(pts1[0], pts1[2])
length = max(width,height)
pts2 = np.array([pts1[0],pts1[0]+np.array([0,length]),
        pts1[0]+np.array([length,0]),pts1[0]+np.array([length,length])]
                ,'float32')

print pts1,pts2

M = cv2.getPerspectiveTransform(pts1,pts2)                   #变换矩阵
new_img = cv2.warpPerspective(img_ori,M,(rows+int(pts1[0,0]),cols+int(pts1[0,1])))  #透视变换

new_img = new_img[max(int(pts1[0,1])-10,0):int(pts1[0,1]+length)+10     #裁剪新图片
            ,max(int(pts1[0,0])-10,0):int(pts1[0,0]+length)+10]


plt.subplot(121),plt.imshow(img_ori)
plt.subplot(122),plt.imshow(new_img)
plt.show()