# -*- coding: utf-8 -*-
# @Time    : 24/11/2016 15:26
# @Author  : Luke
# @Software: PyCharm

from util import *
import matplotlib.pyplot as plt
from itertools import groupby

img_rgb = cv2.imread('cards/test3.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_gray = pre_processing(img_rgb, 0)
r = 1
row_u = 340
row_b = 433

img = img_gray[row_u / r:row_b / r, :]
row, col = img.shape



# sobel,laplacian 边缘检测#
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=5)
laplacian = np.sqrt(sobelx ** 2 + sobely ** 2)
img_e = np.array(laplacian / np.max(laplacian) * 255, 'uint8')
img_g = cv2.GaussianBlur(img_e, (3, 3), 2)
# img_e = img_g

th, img_b = cv2.threshold(img_e, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 用 GaussianBlur的结果比直接的laplacian边界效果好一些

S = np.zeros(img_b.shape, 'uint8')

nb_in = 7
nb_out = 11
l = 3

start = nb_out / 2
row_end = row - start - 1
col_end = col - start - 1
#
for i in range(start, row_end):
    for j in range(start, col_end):
        if img_b[i, j] == 255:
            img_tmp = img_e[i - 3:i + 3 + 1, j - 3:j + 3 + 1]
            P_min = int(img_tmp.min())
            P_max = int(img_tmp.max())
            t = (P_max + P_min) / 2
        else:
            continue
        for i_ in range(i - l, i + l + 1):
            for j_ in range(j - l, j + l + 1):
                if img_e[i_, j_] >= t:
                    S[i_, j_] += 1


plt.subplot(211),plt.imshow(img_b,'gray')
plt.subplot(212),plt.imshow(S,'gray')
plt.show()
