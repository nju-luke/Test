# -*- coding: utf-8 -*-
# @Time    : 25/11/2016 16:24
# @Author  : Luke
# @Software: PyCharm

from util import *
from itertools import groupby
import matplotlib.pyplot as plt

img_rgb = cv2.imread('cards/test3.jpg')


# sobel,laplacian 边缘检测#
def row_crop(img_rgb):
    img_gray = pre_processing(img_rgb, 0)
    r = 1
    row_10 = 340
    row_20 = 433
    img = img_gray[row_10 / r:row_20 / r, :]
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    # laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=5)
    laplacian = np.sqrt(sobelx**2+sobely**2)
    laplacian = np.array(laplacian/np.max(laplacian)*255,'uint8')

    Hy = np.sum(laplacian,axis=1)

    kernal = np.ones(15,'float')
    Hy = np.convolve(Hy,kernal,'same')/len(kernal)
    plt.subplot(211),plt.plot(Hy)

    kernal = np.array([1,-1])
    Hy = np.convolve(Hy,kernal,'same')

    y_index = [id for id in range(len(Hy)-1) if Hy[id]>0 and Hy[id+1]<0]
    distance = [y_index[i] - y_index[i-1] for i in range(1,len(y_index))]
    id = np.argmin([abs(distance[i] - distance[i+1]) if distance[i] > 15 else 20 for i in range(len(distance)-1)])
    half_length = (distance[id] + distance[id+1])/2/2

    row_1 = y_index[id] + row_10 - half_length
    row_2 = y_index[id+2] + row_10 + half_length

    img_new = img_gray[row_1:row_2,:]
    return img_new

img_new = row_crop(img_rgb)

plt.subplot(212),plt.imshow(img_new,'gray')
plt.show()

