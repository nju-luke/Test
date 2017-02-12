# -*- coding: utf-8 -*-
# @Time    : 16/11/2016 09:47
# @Author  : Luke
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('3.jpg',0)
rows ,cols = 1000,649
img = cv2.resize(img,(rows,cols))

lines_result = []

def sobel(length,x_sobel = 0, y_sobel = 0):
    sobel = abs(cv2.Sobel(img, cv2.CV_64F, x_sobel, y_sobel, ksize=5))
    binx = sobel > (np.max(sobel) / 10)
    binx = np.array(binx * 255, 'uint8')
    lines = cv2.HoughLines(binx, 1, np.pi / 180, length / 2)[0]

    min_ = min(lines, key=lambda x: abs(x[0]))
    if min_[0] < length / 10:
        lines_result.append(min_)
    max_ = max(lines, key=lambda x: abs(x[0]))
    if max_[0] > length * 0.9:
        lines_result.append(max_)

    #
    # min_ = min(lines, key=lambda x: abs(x[0]))
    # if min_[0] < length / 10:
    #     lines_result.append(min_)
    # max_ = max(lines, key=lambda x: abs(x[0]))
    # if max_[0] > length * 0.8:
    #     lines_result.append(max_)

sobel(rows,1,0)
sobel(cols,0,1)

for rho,theta in lines_result:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
plt.imshow(img,'gray')
plt.show()