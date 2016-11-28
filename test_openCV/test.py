# -*- coding: utf-8 -*-
# @Time    : 24/11/2016 15:26
# @Author  : Luke
# @Software: PyCharm

from cardNumber.util import *
from itertools import groupby

img_rgb = cv2.imread('cardNumber/cards/test4.jpg')
img_gray = pre_processing(img_rgb,0)
r = 1
row_10 = 340
row_20 = 433
img = img_gray[row_10/r:row_20/r, :]

# sobel,laplacian 边缘检测#
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=5)
laplacian = np.sqrt(sobelx**2+sobely**2)
laplacian = np.array(laplacian/np.max(laplacian)*255,'uint8')

Hy = np.sum(laplacian,axis=1)

kernal = np.ones(10,'float')
Hy = np.convolve(Hy,kernal,'same')/len(kernal)
# plt.subplot(211),plt.plot(Hy)

kernal = np.array([1,-1])
Hy = np.convolve(Hy,kernal,'same')

y_index = [id for id in range(len(Hy)-1) if Hy[id]>0 and Hy[id+1]<0]
distance = [y_index[i] - y_index[i-1] for i in range(1,len(y_index))]
id = np.argmin([abs(distance[i] - distance[i+1]) if distance[i] > 15 else 20 for i in range(len(distance)-1)])
half_length = (distance[id] + distance[id+1])/2/2

row_1 = y_index[id] + row_10 - half_length
row_2 = y_index[id+2] + row_10 + half_length

print row_1,row_2

img = img_gray[row_1:row_2,:]


# 横向切割数字
Hy = np.sum(laplacian,axis=0)

kernal = np.ones(10,'float')
Hy = np.convolve(Hy,kernal,'same')/len(kernal)


# kernal = np.array([1,-1])
# Hy = np.convolve(Hy,kernal,'same')

plt.subplot(211),\
plt.plot(Hy)
plt.subplot(212),\
plt.imshow(img,'gray')
plt.show()

